/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include<omp.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"
#define NUM_GROUP       8

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
} t_param __attribute__((aligned(32)));

typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  propagate;
  cl_kernel  rebound_collision_av;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  cl_mem av_vels;
} t_ocl;

/* struct to hold the 'speed' values */
typedef struct
{
  double speeds[NSPEEDS];
} t_speed __attribute__((aligned(32)));

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr, double** tmp_av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int  timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_ocl ocl,int index,double* av_vels);

//int accelerate_propagate(const t_param params, t_speed* cells, t_speed* tmp_cells,int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, t_ocl ocl);
int rebound_collision_av(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_ocl ocl,int index);
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity_serial(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;    
  t_speed* cells    = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells  = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  double* av_vels  = NULL;     /* a record of the av. velocity computed for each timestep */
  double* tmp_av_vels  = NULL; 
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels,&tmp_av_vels, &ocl);

// Write cells to OpenCL buffer
  err = clEnqueueWriteBuffer(
          ocl.queue, ocl.cells, CL_TRUE, 0,
          sizeof(t_speed) * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);

//               // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
           ocl.queue, ocl.obstacles, CL_TRUE, 0,
           sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  int tot_cells=0; 

  for(int ii=0;ii<params.ny;ii++) {
    for(int jj=0;jj<params.nx;jj++) {
     
      if(!obstacles[ii*params.nx + jj]) {
        tot_cells++; 
      }
    }
  }
 
  // Write cells to OpenCL buffer
 // err = clEnqueueWriteBuffer(
  //  ocl.queue, ocl.cells, CL_TRUE, 0,
  //  sizeof(t_speed) * params.nx * params.ny, cells, 0, NULL, NULL);
 // checkError(err, "writing cells data", __LINE__);

  // Write obstacles to OpenCL buffer
 // err = clEnqueueWriteBuffer(
  //  ocl.queue, ocl.obstacles, CL_TRUE, 0,
   // sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  //checkError(err, "writing obstacles data", __LINE__);
  

  
  #pragma vector aligned
  for (unsigned int tt=0; tt < params.maxIters;tt++ )
  {
   timestep(params, cells, tmp_cells, obstacles, ocl, tt, av_vels);

//#ifdef DEBUG
   // printf("==timestep: %d==\n", tt);
   // printf("av velocity: %.12E\n", av_vels[tt]);
   // printf("tot density: %.12E\n", total_density(params, cells));
//#endif
  }
 err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(t_speed) * params.nx * params.ny, cells, 0, NULL, NULL);
    checkError(err, "writing cells data", __LINE__);

/* for(int i = 0;i<16;i++)
 {
    err = clEnqueueReadBuffer(
       ocl.queue, ocl.av_vels, CL_TRUE, 0, 
       sizeof(cl_double) * params.maxIters , av_vels, 0, NULL, NULL);
    checkError(err, "reading av_vels data", __LINE__);

 }*/
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.av_vels, CL_TRUE, 0,
    sizeof(cl_double) * params.maxIters * ((params.nx/NUM_GROUP)*(params.ny/NUM_GROUP)) , tmp_av_vels, 0, NULL, NULL);
  checkError(err, "reading av_vels data", __LINE__);
 
  for (unsigned int tt=0; tt < params.maxIters;tt++ )
  {
      double av = 0.0;
      for(int i = 0;i<((params.nx/NUM_GROUP)*(params.ny/NUM_GROUP));i++)
      {
         av +=  tmp_av_vels[tt*((params.nx/NUM_GROUP)*(params.ny/NUM_GROUP)) + i];
      }
      av_vels[tt] = av/(double)tot_cells;

  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

/*      timestep fucntion     */
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_ocl ocl,int index, double* av_vels)
{
  cl_int err;
 // Write cells to device
 /* err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(t_speed) * params.nx * params.ny, cells, 0, NULL, NULL);
    checkError(err, "writing cells data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.tmp_cells, CL_TRUE, 0,
    sizeof(t_speed) * params.nx * params.ny, tmp_cells, 0, NULL, NULL);
  checkError(err, "reading tmp_cells data", __LINE__);
*/
  accelerate_flow(params, cells, obstacles, ocl);
  propagate(params, cells, tmp_cells, ocl);
  rebound_collision_av(params,cells,tmp_cells,obstacles,ocl,index);
  // Read tmp_cells from device
 // err = clEnqueueReadBuffer(
   // ocl.queue, ocl.tmp_cells, CL_TRUE, 0,
   // sizeof(t_speed) * params.nx * params.ny, tmp_cells, 0, NULL, NULL);
  //checkError(err, "reading tmp_cells data", __LINE__);

 // rebound_collision_av(params,cells,tmp_cells,obstacles,ocl,index);
  

  return EXIT_SUCCESS;
}


int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
  /* compute weighting factors */
  //double w1 = params.density * params.accel / 9.0;
  //double w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  //int ii = params.ny - 2;
  //#pragma vector aligned
  //for (int jj = 0; jj < params.nx; jj++)
  //{
    /* if the cell is not occupied and
 *     ** we don't send a negative density */
    //if (!obstacles[ii * params.nx + jj]
    //   && (cells[ii * params.nx + jj].speeds[3] - w1) > 0.0
    //    && (cells[ii * params.nx + jj].speeds[6] - w2) > 0.0
    //   && (cells[ii * params.nx + jj].speeds[7] - w2) > 0.0)
    //{
      /* increase 'east-side' densities */
    //  cells[ii * params.nx + jj].speeds[1] += w1;
    //  cells[ii * params.nx + jj].speeds[5] += w2;
     // cells[ii * params.nx + jj].speeds[8] += w2;
      /* decrease 'west-side' densities */
     // cells[ii * params.nx + jj].speeds[3] -= w1;
     // cells[ii * params.nx + jj].speeds[6] -= w2;
     // cells[ii * params.nx + jj].speeds[7] -= w2;
    //}
  //}
   cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_double), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_double), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;

}
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, t_ocl ocl)
{
  /* loop over _all_ cells */
  /*#pragma vector aligned
  #pragma omp parallel for
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {*/
      /* determine indices of axis-direction neighbours
 *       ** respecting periodic boundary conditions (wrap around) */
      /*int y_n = (ii + 1) % params.ny;
      int x_e = (jj + 1) % params.nx;
      int y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
      int x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);*/
      /* propagate densities to neighbouring cells, following
 *       ** appropriate directions of travel and writing into
 *             ** scratch space grid */
      //tmp_cells[ii * params.nx + jj].speeds[0]  = cells[ii * params.nx + jj].speeds[0]; /* central cell, no movement */
      //tmp_cells[ii * params.nx + jj].speeds[1] = cells[ii * params.nx + x_w].speeds[1]; /* east */
      //tmp_cells[ii * params.nx + jj].speeds[2]  = cells[y_s * params.nx + jj].speeds[2]; /* north */
      //tmp_cells[ii * params.nx + jj].speeds[3] = cells[ii * params.nx + x_e].speeds[3]; /* west */
      //tmp_cells[ii * params.nx + jj].speeds[4]  = cells[y_n * params.nx + jj].speeds[4]; /* south */
      //tmp_cells[ii * params.nx + jj].speeds[5] = cells[y_s * params.nx + x_w].speeds[5]; /* north-east */
      //tmp_cells[ii * params.nx + jj].speeds[6] = cells[y_s * params.nx + x_e].speeds[6]; /* north-west */
      //tmp_cells[ii * params.nx + jj].speeds[7] = cells[y_n * params.nx + x_e].speeds[7]; /* south-west */
      //tmp_cells[ii * params.nx + jj].speeds[8] = cells[y_n * params.nx + x_w].speeds[8]; /* south-east */
    //}
  //}
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting propagate arg 0", __LINE__);
  err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting propagate arg 1", __LINE__);
  err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting propagate arg 2", __LINE__);
  err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_int), &params.nx);
  checkError(err, "setting propagate arg 3", __LINE__);
  err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_int), &params.ny);
  checkError(err, "setting propagate arg 4", __LINE__);

  // Enqueue kernel
  size_t global[2] = {params.nx, params.ny};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.propagate,
                               2, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing propagate kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for propagate kernel", __LINE__);

  return EXIT_SUCCESS;

}


/*          Combining rebound, collision and av_velocity fucntions       */
int rebound_collision_av(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_ocl ocl,int index)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.rebound_collision_av, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting rebound_collision_av arg 0", __LINE__);

  err = clSetKernelArg(ocl.rebound_collision_av, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting rebound_collision_av arg 1", __LINE__);

  err = clSetKernelArg(ocl.rebound_collision_av, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting rebound_collision_av arg 2", __LINE__);

  err = clSetKernelArg(ocl.rebound_collision_av, 3, sizeof(cl_mem), &ocl.av_vels);
  checkError(err, "setting rebound_collision_av arg 3", __LINE__);


  err = clSetKernelArg(ocl.rebound_collision_av, 4, sizeof(cl_int), &params.nx);
  checkError(err, "setting rebound_collision_av arg 4", __LINE__);

  err = clSetKernelArg(ocl.rebound_collision_av, 5, sizeof(cl_int), &params.ny);
  checkError(err, "setting rebound_collision_av arg 5", __LINE__);

  err = clSetKernelArg(ocl.rebound_collision_av, 6, sizeof(cl_double), &params.omega);
  checkError(err, "setting rebound_collision_av arg 6", __LINE__); 

  err = clSetKernelArg(ocl.rebound_collision_av, 7, sizeof(cl_double)*NUM_GROUP*NUM_GROUP, NULL);
  checkError(err, "setting rebound_collision_av arg 7", __LINE__);

 // err = clSetKernelArg(ocl.rebound_collision_av, 8, sizeof(cl_int)*32*32, NULL);
 // checkError(err, "setting rebound_collision_av arg 8", __LINE__);


  err = clSetKernelArg(ocl.rebound_collision_av, 8, sizeof(cl_int), &index);
  checkError(err, "setting rebound_collision_av arg 8", __LINE__);     
  
  // Enqueue kernel
  size_t global[2] = {params.nx, params.ny};
  size_t local[2] = {NUM_GROUP,NUM_GROUP};
 // size_t local[1]={512};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.rebound_collision_av,
                               2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing rebound_collision_av kernel", __LINE__);
  // print("%d",err); 
  return EXIT_SUCCESS;
  		
  /***********************************************************/	
  //const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  //const double w0 = 4.0 / 9.0;  /* weighting factor */
 // const double w1 = 1.0 / 9.0;  /* weighting factor */
  //const double w2 = 1.0 / 36.0; /* weighting factor */
 // const double res1 = 2.0 / 9.0;
  //const double res2 =2.0 / 3.0;
  //unsigned int tot_cells = 0;  /* no. of cells used in calculation */
  //double tot_u = 0.0;
  
  /* loop over the cells in the grid
 *   ** NB the collision step is called after
 *     ** the propagate step and so values of interest
 *       ** are in the scratch-space grid */
 //#pragma vector aligned
 //#pragma omp parallel for schedule(static) reduction(+: tot_u,tot_cells)
  /*for (unsigned int ii =0;ii<params.ny; ii++  )
  {
    for (unsigned int jj=0; jj<params.nx; jj++  )
    {
      
      unsigned int index = ii * params.nx + jj;*/
        /*   rebound function    */
      //if (obstacles[index])
     // {
        /* called after propagate, so taking values from scratch space
 *  *         ** mirroring, and writing into main grid */
        // memcpy(&(cells[index].speeds[1]),&(tmp_cells[index].speeds[3]),sizeof(t_speed)/4.5);
        // memcpy(&(cells[index].speeds[3]),&(tmp_cells[index].speeds[1]),sizeof(t_speed)/4.5);
       //  memcpy(&(cells[index].speeds[5]),&(tmp_cells[index].speeds[7]),sizeof(t_speed)/4.5);
        // memcpy(&(cells[index].speeds[7]),&(tmp_cells[index].speeds[5]),sizeof(t_speed)/4.5);
         
   //   }

      /*     collision function     */
     // else
      //{
        /* compute local density total */
       // double local_density = 0.0;
        /* called after propagate, so taking values from scratch space
 *  *         ** mirroring, and writing into main grid */
        /*local_density += tmp_cells[index].speeds[0];
        local_density += tmp_cells[index].speeds[1];
        local_density += tmp_cells[index].speeds[2];
        local_density += tmp_cells[index].speeds[3];
        local_density += tmp_cells[index].speeds[4];
        local_density += tmp_cells[index].speeds[5];
        local_density += tmp_cells[index].speeds[6];
        local_density += tmp_cells[index].speeds[7];
        local_density += tmp_cells[index].speeds[8];*/
          
        
        /* compute x velocity component */
       /* double u0 = (tmp_cells[index].:speeds[1]
                   + tmp_cells[index].speeds[5]
                   + tmp_cells[index].speeds[8]
                   - (tmp_cells[index].speeds[3]
                   + tmp_cells[index].speeds[6]
                   + tmp_cells[index].speeds[7]));

        double u_x = u0 / local_density;*/
        /* compute y velocity component */
       /* double u1 = (tmp_cells[index].speeds[2]
                   + tmp_cells[index].speeds[5]
                   + tmp_cells[index].speeds[6]
                    - (tmp_cells[index].speeds[4]
                   + tmp_cells[index].speeds[7]
                   + tmp_cells[index].speeds[8]));

        double u_y = u1 /local_density ;*/


        /* velocity squared */
        //double u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        //double u[NSPEEDS];
        //u[1] = u_x * u_x;
        //u[2] = u_y * u_y;
        //u[5] =   u_x + u_y;  /* north-east */
        //u[6] = - u_x + u_y;  /* north-west */
        //u[7] = - u[5];  /* south-west */
        //u[8] =  -u[6];  /* south-east */
       
        /* equilibrium densities */
       /* double d_equ[NSPEEDS];
        double t0 = w0 * local_density;
        double t1 = w1 * local_density;
        double t2 = w2 * local_density;
        d_equ[0] = t0 * (1.0 - u_sq / res2);*/
        /* axis speeds: weight w1 */
        /*d_equ[1] = t1 * (1.0 + u_x / c_sq
                    + u[1] / res1
                    - u_sq / res2);
        d_equ[2] = t1 * (1.0 + u_y / c_sq
                    + u[2]  / res1
                    - u_sq / res2);
        d_equ[3] = t1 * (1.0 +(- u_x) / c_sq
                    + u[1] / res1
                    - u_sq / res2);
        d_equ[4] = t1 * (1.0 +(- u_y) / c_sq
                    + u[2] / res1
                    - u_sq / res2);*/
        /* diagonal speeds: weight w2 */
        /*d_equ[5] = t2 * (1.0 + u[5] / c_sq
                    + (u[5]*u[5]) / res1
                    - u_sq / res2);
        d_equ[6] = t2 * (1.0 + u[6] / c_sq
                    + (u[6]*u[6]) / res1
                    - u_sq / res2);
        d_equ[7] = t2 * (1.0 + u[7] / c_sq
                    + (u[7]*u[7]) / res1
                    - u_sq / res2);
        d_equ[8] = t2 * (1.0 + u[8] / c_sq
                    + (u[8]*u[8]) / res1
                    - u_sq / res2);*/

        /* relaxation step */
        /*cells[index].speeds[0] = tmp_cells[index].speeds[0] + params.omega
                                * (d_equ[0] - tmp_cells[index].speeds[0]);
                                 
        cells[index].speeds[1] = tmp_cells[index].speeds[1] + params.omega
                                 * (d_equ[1] - tmp_cells[index].speeds[1]);
                                 
        cells[index].speeds[2] = tmp_cells[index].speeds[2] + params.omega
                                 * (d_equ[2] - tmp_cells[index].speeds[2]);

        cells[index].speeds[3] = tmp_cells[index].speeds[3] + params.omega
                                 * (d_equ[3] - tmp_cells[index].speeds[3]);                                                                                            cells[index].speeds[4] = tmp_cells[index].speeds[4] + params.omega                                       * (d_equ[4] - tmp_cells[index].speeds[4]);

        cells[index].speeds[5] = tmp_cells[index].speeds[5] + params.omega
                                * (d_equ[5] - tmp_cells[index].speeds[5]);

        cells[index].speeds[6] = tmp_cells[index].speeds[6] + params.omega
                                 * (d_equ[6] - tmp_cells[index].speeds[6]);
                                 
        cells[index].speeds[7] = tmp_cells[index].speeds[7] + params.omega
                                 * (d_equ[7] - tmp_cells[index].speeds[7]);
                                 
        cells[index].speeds[8] = tmp_cells[index].speeds[8] + params.omega
                                 * (d_equ[8] - tmp_cells[index].speeds[8]);

        double  value = (u_x * u_x)+(u_y * u_y);*/
        /* accumulate the norm of x- and y- velocity components */
        //tot_u += sqrt(value);
        /* increase counter of inspected cells */
        //++tot_cells;
     // }                                                                             } 
  //}
 // return tot_u/(double)tot_cells;
                                 
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        double local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* x-component of velocity */
        double u_x = (cells[ii * params.nx + jj].speeds[1]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[8]
                      - (cells[ii * params.nx + jj].speeds[3]
                         + cells[ii * params.nx + jj].speeds[6]
                         + cells[ii * params.nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        double u_y = (cells[ii * params.nx + jj].speeds[2]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[6]
                      - (cells[ii * params.nx + jj].speeds[4]
                         + cells[ii * params.nx + jj].speeds[7]
                         + cells[ii * params.nx + jj].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (double)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr, double** tmp_av_vels_ptr, t_ocl *ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */
  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0 / 9.0;
  float w1 = params->density      / 9.0;
  float w2 = params->density      / 36.0;
  unsigned int ii = 0;
  unsigned int jj = 0;
  unsigned int index=0;

  for ( ii = 0; ii < params->ny; ii++)
  {
    for (  jj = 0; jj < params->nx; jj++)
    {
      index = ii * params->nx + jj;
      /* centre */
      (*cells_ptr)[index].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[index].speeds[1] = w1;
      (*cells_ptr)[index].speeds[2] = w1;
      (*cells_ptr)[index].speeds[3] = w1;
      (*cells_ptr)[index].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[index].speeds[5] = w2;
      (*cells_ptr)[index].speeds[6] = w2;
      (*cells_ptr)[index].speeds[7] = w2;
      (*cells_ptr)[index].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for ( unsigned ii = 0; ii < params->ny; ii++)
  {
    for ( unsigned jj = 0; jj < params->nx; jj++)
    {
      index = ii * params->nx + jj;

      (*obstacles_ptr)[index] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);
  *tmp_av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters * ((params->nx/NUM_GROUP)*(params->ny/NUM_GROUP)));

  
  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->propagate = clCreateKernel(ocl->program, "propagate", &err);
  checkError(err, "creating propagate kernel", __LINE__);
  ocl->rebound_collision_av = clCreateKernel(ocl->program, "rebound_collision_av", &err);
  checkError(err, "creating rebound_collision_av kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_speed) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_speed) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);
  ocl->av_vels = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_double) * params->maxIters * ((params->nx/NUM_GROUP)*(params->ny/NUM_GROUP)), NULL, &err);
  checkError(err, "creating av_vels buffer", __LINE__);

  
  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;
  
  
  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseMemObject(ocl.av_vels);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.propagate);
  clReleaseKernel(ocl.rebound_collision_av);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}



double calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  double total = 0.0;  /* accumulator */

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      unsigned int index = ii * params.nx + jj;
      for (unsigned int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[index].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      unsigned int index = ii * params.nx + jj;
      /* an occupied cell */
      if (obstacles[index])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[index].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[index].speeds[1]
               + cells[index].speeds[5]
               + cells[index].speeds[8]
               - (cells[index].speeds[3]
                  + cells[index].speeds[6]
                  + cells[index].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[index].speeds[2]
               + cells[index].speeds[5]
               + cells[index].speeds[6]
               - (cells[index].speeds[4]
                  + cells[index].speeds[7]
                  + cells[index].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[index]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (unsigned int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}
void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}
void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
