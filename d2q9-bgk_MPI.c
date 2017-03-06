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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER          0   

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param __attribute__((aligned(32)));
      

/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} t_speed __attribute__((aligned(32)));

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
          t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, 
           int** obstacles_ptr, float** av_vels_ptr);

/* 
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate() & collision()
*/
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int start_point, int end_point);
int halo_propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, int start_point, int end_point,int index);
int collision_rebound_av_velocity(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* av_vels, int index, int start_point, int end_point);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
         int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, float final_average_velocity);

/* utility functions */
void die(const char* message, const int line, const char *file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speed* cells     = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
    int      ii;                  /* counter */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
    double usrtim;                /* floating point number to record elapsed user CPU time */
    double systim;                /* floating point number to record elapsed system CPU time */

    /* parse the command line */
    if(argc != 3) 
    {
        usage(argv[0]);
    } 
    else 
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* initialise our data structures and load values from file */
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

    // Initialize MPI environment.
    MPI_Init(&argc, &argv);

    int flag;
    // Check if initialization was successful.
    MPI_Initialized(&flag);
    if(flag != 1)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank;  /*rank number*/
    int size;  /*size number*/
    int jundgement;    
    int packet_size; /*size of data*/
    int buff_size;  /* size of update data*/
    int start_point; /*for-loop'starting*/
    int end_point;   /*for-loop'end*/   

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    jundgement = params.ny%size;
    packet_size = (rank<jundgement)?(params.ny/size + 1) * params.nx:(params.ny/size) * params.nx;

    if(jundgement == 0) 
    {
    	buff_size = packet_size;
    	start_point = packet_size * rank;
        end_point = packet_size * (rank+1);
    } 
    else if(rank < jundgement) 
    {
        buff_size = packet_size;
	start_point = packet_size * rank;
        end_point = packet_size * (rank+1);
    } 
    else 
    {
    	buff_size = (params.ny/size + 1) * params.nx;
        start_point = (packet_size + params.nx) * jundgement + packet_size * (rank-jundgement);
        end_point = (packet_size + params.nx) * jundgement + packet_size * (rank-jundgement+1);
    }
    
    /* iterate for maxIters timesteps */
    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    #pragma vector aligned
    for(ii=0;ii<params.maxIters;ii++) 
    {
        accelerate_flow(params,cells,obstacles, start_point, end_point);
        halo_propagate(params,cells,tmp_cells, start_point, end_point,ii);
        collision_rebound_av_velocity(params,cells,tmp_cells,obstacles, av_vels, ii, start_point, end_point);   
        
        /*#ifdef DEBUG
        printf("==timestep: %d==\n",ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n",total_density(params,cells));
        #endif*/
    }

    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr=ru.ru_utime;        
    usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    timstr=ru.ru_stime;        
    systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    float* buffer = malloc(buff_size * 9 * sizeof(float));


    if(rank != MASTER) {
        for(ii=0;ii<packet_size;ii++) {
            int index1 = 9*ii;
            int index2 = start_point+ii;
            buffer[index1]   = cells[index2].speeds[0];
            buffer[index1+1] = cells[index2].speeds[1];
            buffer[index1+2] = cells[index2].speeds[2];
            buffer[index1+3] = cells[index2].speeds[3];
            buffer[index1+4] = cells[index2].speeds[4];
            buffer[index1+5] = cells[index2].speeds[5];
            buffer[index1+6] = cells[index2].speeds[6];
            buffer[index1+7] = cells[index2].speeds[7];
            buffer[index1+8] = cells[index2].speeds[8];
        }
        MPI_Ssend(buffer, 9*buff_size, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
    } 
    else 
    {
       MPI_Status status;
       int jj;
      // int index2 = 9*jj;
       if(jundgement == 0)
       {
         for(ii=1;ii<size;ii++) 
         {
            MPI_Recv(buffer, 9*buff_size, MPI_FLOAT, ii, 3, MPI_COMM_WORLD, &status);
            for(jj=0;jj<packet_size;jj++) {
                int index1 = packet_size*ii+jj;
                int index2 = 9 * jj;
				cells[index1].speeds[0] = buffer[index2];
				cells[index1].speeds[1] = buffer[index2+1];
				cells[index1].speeds[2] = buffer[index2+2];
				cells[index1].speeds[3] = buffer[index2+3];
				cells[index1].speeds[4] = buffer[index2+4];
				cells[index1].speeds[5] = buffer[index2+5];
				cells[index1].speeds[6] = buffer[index2+6];
				cells[index1].speeds[7] = buffer[index2+7];
				cells[index1].speeds[8] = buffer[index2+8];
				}
          }
       } 
       else 
       {
          for(ii=1;ii<jundgement;ii++) {
              MPI_Recv(buffer, 9*buff_size, MPI_FLOAT, ii, 3, MPI_COMM_WORLD, &status);
	      for(jj=0;jj<packet_size;jj++) {
              int index1 = packet_size*ii+jj;
              int index2 = 9 * jj;
			  cells[index1].speeds[0] = buffer[index2];
			  cells[index1].speeds[1] = buffer[index2+1];
			  cells[index1].speeds[2] = buffer[index2+2];
		      cells[index1].speeds[3] = buffer[index2+3];
		      cells[index1].speeds[4] = buffer[index2+4];
		      cells[index1].speeds[5] = buffer[index2+5];
		      cells[index1].speeds[6] = buffer[index2+6];
		      cells[index1].speeds[7] = buffer[index2+7];
		      cells[index1].speeds[8] = buffer[index2+8];
	      }
        }
           for(ii=jundgement;ii<size;ii++) {
	       MPI_Recv(buffer, 9*buff_size, MPI_FLOAT, ii, 3, MPI_COMM_WORLD, &status);
              for(jj=0;jj<(params.ny/size) * params.nx;jj++) {
                    int index1 = packet_size * jundgement + (packet_size-params.nx) * (ii-jundgement) + jj;
					int index2 = 9*jj;
                    cells[index1].speeds[0] = buffer[index2];
                    cells[index1].speeds[1] = buffer[index2+1];
                    cells[index1].speeds[2] = buffer[index2+2];
					cells[index1].speeds[3] = buffer[index2+3];
					cells[index1].speeds[4] = buffer[index2+4];
					cells[index1].speeds[5] = buffer[index2+5];
					cells[index1].speeds[6] = buffer[index2+6];
					cells[index1].speeds[7] = buffer[index2+7];
					cells[index1].speeds[8] = buffer[index2+8];
			}
          }
        }
        free(buffer);
        buffer = NULL;
    }

    // Finalize MPI environment.
    MPI_Finalize();

    MPI_Finalized(&flag);
    if(flag != 1) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if(rank == 0) {
        /* write final values and free memory */
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n",calc_reynolds(params,cells,obstacles,av_vels[params.maxIters-1]));
        printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
        printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
        printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
        write_values(params,cells,obstacles,av_vels);
        finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
    }
    
    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int start_point, int end_point)
{
    int ii;     /* generic counters */
    int size;
    /* compute weighting factors */
    float w2 = params.density * params.accel * 0.02777777777;
    float w1 = w2 * 4;

    /* modify the 2nd row of the grid */
     #pragma vector aligned
    for(ii=(params.ny*params.nx - 2*params.nx);ii<(params.ny*params.nx - params.nx);ii++) {
        /* if the cell is not occupied and
        ** we don't send a density negative */
        if( !obstacles[ii] && 
        (cells[ii].speeds[3] - w1) > 0.0 &&
        (cells[ii].speeds[6] - w2) > 0.0 &&
        (cells[ii].speeds[7] - w2) > 0.0 ) {
            /* increase 'east-side' densities */
            cells[ii].speeds[1] += w1;
            cells[ii].speeds[5] += w2;
            cells[ii].speeds[8] += w2;
            /* decrease 'west-side' densities */
            cells[ii].speeds[3] -= w1;
            cells[ii].speeds[6] -= w2;
            cells[ii].speeds[7] -= w2;
        }
    }   
    return EXIT_SUCCESS;
}

int halo_propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, int start_point, int end_point,int index)
{
    int i;
    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    /*****************************************/
    int buffer_size = params.nx * 3;
    float* to_up_array = malloc(sizeof(float) * buffer_size);
    float* to_down_array = malloc(sizeof(float) * buffer_size);
    float* from_up_array = malloc(sizeof(float) * buffer_size);
    float* from_down_array = malloc(sizeof(float) * buffer_size);
    MPI_Status status; 
    // MPI_Request sendRequest1,recvRequest1,sendRequest2,recvRequest2; 
     if(index != 0 ) {
       // int rank; 
        int up_index; 
        int down_index;

        // Computing sending index and receiving index.
        if(rank == MASTER) 
        {
            up_index = 1;
            down_index = size-1;
        } 
        else if(rank == size-1) 
        {
            up_index = 0;
            down_index = size-2;
        }
        else
        {
            up_index = rank+1;
            down_index = rank-1;
        }

         // MPI_Irecv(from_down_array, buffer_size, MPI_FLOAT, down_index, 2, MPI_COMM_WORLD, &recvRequest1);
         // MPI_Irecv(from_up_array, buffer_size, MPI_FLOAT, up_index, 1, MPI_COMM_WORLD, &recvRequest2);
        
       
        // Copy values to be sent.
        #pragma vector aligned
        for(i=0; i<params.nx; i++) {
            int index1 = i*3;
            int index2 = start_point+i;
            int index3 = end_point-params.nx+i;
            to_down_array[index1]    = cells[index2].speeds[4];
            to_down_array[index1+1]  = cells[index2].speeds[7];
            to_down_array[index1+2]  = cells[index2].speeds[8];

            to_up_array[index1]      = cells[index3].speeds[2];
            to_up_array[index1+1]    = cells[index3].speeds[5];
            to_up_array[index1+2]    = cells[index3].speeds[6];
        } 
         // MPI_Irecv(from_down_array, buffer_size, MPI_FLOAT, down_index, 2, MPI_COMM_WORLD, &recvRequest1);
         // MPI_Irecv(from_up_array, buffer_size, MPI_FLOAT, up_index, 1, MPI_COMM_WORLD, &recvRequest1);
        //  MPI_Isend(to_up_array, buffer_size, MPI_FLOAT, up_index, 2, MPI_COMM_WORLD,&sendRequest2);
         // MPI_Isend(to_down_array, buffer_size, MPI_FLOAT, down_index, 1, MPI_COMM_WORLD,&sendRequest1);
        
         /* send down, receive from up */
         MPI_Sendrecv(to_down_array, params.nx * 3, MPI_FLOAT, down_index, 1,
                    from_up_array, params.nx * 3, MPI_FLOAT, up_index, 1,
                    MPI_COMM_WORLD, &status);

        // send up, receive from down 
         MPI_Sendrecv(to_up_array, params.nx * 3, MPI_FLOAT, up_index, 2,
                     from_down_array, params.nx * 3, MPI_FLOAT, down_index, 2,
                     MPI_COMM_WORLD, &status);
       
        // MPI_Wait(&recvRequest1, &status);
        // MPI_Wait(&recvRequest2, &status);
                   
        if(rank == MASTER) {
            // From up copy on row above.
            // From down copy on last row.
            #pragma vector aligned
            for(i=0; i<params.nx; i++) {
                int index1 = end_point+i;
                int index2 = params.nx*(params.ny-1)+i;
                int index3 = i*3;
                cells[index1].speeds[4] = from_up_array[index3];
                cells[index1].speeds[7] = from_up_array[index3+1];
                cells[index1].speeds[8] = from_up_array[index3+2];
                // Rank 0 copies on last row of cells
                cells[index2].speeds[2] = from_down_array[index3];
                cells[index2].speeds[5] = from_down_array[index3+1];
                cells[index2].speeds[6] = from_down_array[index3+2];
            }
        } else if(rank == size-1) {
            // From up copy on first row.
            // From down copy on row below.
            #pragma vector aligned
            for(i=0; i<params.nx; i++) {
               	int index1 = i*3;
 		int index2 = start_point-params.nx+i;
                cells[i].speeds[4] = from_up_array[index1];
                cells[i].speeds[7] = from_up_array[index1+1];
                cells[i].speeds[8] = from_up_array[index1+2];

                cells[index2].speeds[2] = from_down_array[index1];
                cells[index2].speeds[5] = from_down_array[index1+1];
                cells[index2].speeds[6] = from_down_array[index1+2];
            }
        } else {
            #pragma vector aligned
            for(i=0; i<params.nx; i++) {
                int index1 = end_point+i;
                int index2 = start_point-params.nx+i;
                int index3 = i*3;
                cells[index1].speeds[4] = from_up_array[index3];
                cells[index1].speeds[7] = from_up_array[index3+1];
                cells[index1].speeds[8] = from_up_array[index3+2];

                cells[index2].speeds[2] = from_down_array[index3];
                cells[index2].speeds[5] = from_down_array[index3+1];
                cells[index2].speeds[6] = from_down_array[index3+2];
            }
        }
        
       
      }
    /*****************************************/
    // First process gets first line.
    if(rank == MASTER) {
        #pragma vector aligned
        for(i=1;i<params.nx-1;i++) {
            int index = params.nx*(params.ny-1)+i;
            int index1 = i+params.nx;
            tmp_cells[i].speeds[0] = cells[i].speeds[0];
            tmp_cells[i].speeds[1] = cells[i-1].speeds[1];
            tmp_cells[i].speeds[2] = cells[index].speeds[2];
            tmp_cells[i].speeds[3] = cells[i+1].speeds[3];
            tmp_cells[i].speeds[4] = cells[index1].speeds[4];
            tmp_cells[i].speeds[5] = cells[index-1].speeds[5];
            tmp_cells[i].speeds[6] = cells[index+1].speeds[6];
            tmp_cells[i].speeds[7] = cells[index1+1].speeds[7];
            tmp_cells[i].speeds[8] = cells[index1-1].speeds[8];
        }
       // MPI_Wait(&recvRequest1, &status);
       // MPI_Wait(&recvRequest2, &status);
       // MPI_Wait(&sendRequest1, &status);
       // MPI_Wait(&sendRequest2, &status);
                        
        // Lower left corner
        tmp_cells[0].speeds[0] = cells[0].speeds[0];
        tmp_cells[0].speeds[1] = cells[params.nx-1].speeds[1];
        tmp_cells[0].speeds[2] = cells[params.nx*(params.ny-1)].speeds[2];
        tmp_cells[0].speeds[3] = cells[1].speeds[3];
        tmp_cells[0].speeds[4] = cells[params.nx].speeds[4];
        tmp_cells[0].speeds[5] = cells[params.nx*params.ny-1].speeds[5];
        tmp_cells[0].speeds[6] = cells[params.nx*(params.ny-1)+1].speeds[6];
        tmp_cells[0].speeds[7] = cells[params.nx+1].speeds[7];
        tmp_cells[0].speeds[8] = cells[2*params.nx-1].speeds[8];
      //  MPI_Wait(&recvRequest1, &status);
      //  MPI_Wait(&sendRequest2, &status);
        // Lower right corner
        int index = params.nx-1;
        tmp_cells[index].speeds[0] = cells[params.nx-1].speeds[0];
        tmp_cells[index].speeds[1] = cells[params.nx-2].speeds[1];
        tmp_cells[index].speeds[2] = cells[(params.nx-1)*(params.ny-1)].speeds[2];
        tmp_cells[index].speeds[3] = cells[0].speeds[3];
        tmp_cells[index].speeds[4] = cells[2*params.nx-1].speeds[4];
        tmp_cells[index].speeds[5] = cells[params.nx*params.ny-2].speeds[5];
        tmp_cells[index].speeds[6] = cells[params.nx*(params.ny-1)].speeds[6];
        tmp_cells[index].speeds[7] = cells[params.nx].speeds[7];
        tmp_cells[index].speeds[8] = cells[2*params.nx-2].speeds[8];

       
    }

    // Last process gets last line.
    if(rank == size-1) {
        #pragma vector aligned
        for(i=params.nx*params.ny-params.nx+1;i<params.ny*params.nx-1;i++) {
            tmp_cells[i].speeds[0] = cells[i].speeds[0];
            tmp_cells[i].speeds[1] = cells[i-1].speeds[1];
            tmp_cells[i].speeds[2] = cells[i-params.nx].speeds[2];
            tmp_cells[i].speeds[3] = cells[i+1].speeds[3];
            tmp_cells[i].speeds[4] = cells[i-params.nx*(params.ny-1)].speeds[4];
            tmp_cells[i].speeds[5] = cells[i-params.nx-1].speeds[5];
            tmp_cells[i].speeds[6] = cells[i-params.nx+1].speeds[6];
            tmp_cells[i].speeds[7] = cells[i-params.nx*(params.ny-1)+1].speeds[7];
            tmp_cells[i].speeds[8] = cells[i-params.nx*(params.ny-1)-1].speeds[8];
        }
       // MPI_Wait(&recvRequest, &status);
      //  MPI_Wait(&sendRequest, &status);
      

        // Upper left corner
        int index1 = params.nx*(params.ny-1);
        int index2 = params.nx*params.ny-1;
        tmp_cells[index1].speeds[0] = cells[index1].speeds[0];
        tmp_cells[index1].speeds[1] = cells[params.nx*params.ny-1].speeds[1];
        tmp_cells[index1].speeds[2] = cells[params.nx*(params.ny-2)].speeds[2];
        tmp_cells[index1].speeds[3] = cells[params.nx*(params.ny-1)+1].speeds[3];
        tmp_cells[index1].speeds[4] = cells[0].speeds[4];
        tmp_cells[index1].speeds[5] = cells[params.nx*(params.ny-1)-1].speeds[5];
        tmp_cells[index1].speeds[6] = cells[params.nx*(params.ny-2)+1].speeds[6];
        tmp_cells[index1].speeds[7] = cells[1].speeds[7];
        tmp_cells[index1].speeds[8] = cells[params.nx-1].speeds[8];
    //      MPI_Wait(&recvRequest1, &status);
      //  MPI_Wait(&sendRequest2, &status);

        // Upper right corner
        tmp_cells[index2].speeds[0] = cells[index2].speeds[0];
        tmp_cells[index2].speeds[1] = cells[params.nx*params.ny-2].speeds[1];
        tmp_cells[index2].speeds[2] = cells[params.nx*(params.ny-1)-1].speeds[2];
        tmp_cells[index2].speeds[3] = cells[params.nx*(params.ny-1)].speeds[3];
        tmp_cells[index2].speeds[4] = cells[params.nx-1].speeds[4];
        tmp_cells[index2].speeds[5] = cells[params.nx*(params.ny-1)-2].speeds[5];
        tmp_cells[index2].speeds[6] = cells[params.nx*(params.ny-2)].speeds[6];
        tmp_cells[index2].speeds[7] = cells[0].speeds[7];
        tmp_cells[index2].speeds[8] = cells[params.nx-2].speeds[8];

        
    }

    if(rank == MASTER) {
        start_point = params.nx;
    }
    if(rank == size-1) {
    	end_point = end_point-params.nx;
    }

    //for(ii=params.ny; ii<params.nx*(params.ny-1); ii++) {
    #pragma vector aligned
    for(i=start_point; i<end_point; i++) {
        tmp_cells[i].speeds[0] = cells[i].speeds[0];
        tmp_cells[i].speeds[1] = cells[i-1].speeds[1];
        tmp_cells[i].speeds[2] = cells[i-params.nx].speeds[2];
        tmp_cells[i].speeds[3] = cells[i+1].speeds[3];
        tmp_cells[i].speeds[4] = cells[i+params.nx].speeds[4];
        tmp_cells[i].speeds[5] = cells[i-params.nx-1].speeds[5];
        tmp_cells[i].speeds[6] = cells[i-params.nx+1].speeds[6];
        tmp_cells[i].speeds[7] = cells[i+params.nx+1].speeds[7];
        tmp_cells[i].speeds[8] = cells[i+params.nx-1].speeds[8];
       // MPI_Wait(&recvRequest, &status);
       // MPI_Wait(&sendRequest, &status);
       
        // First column
        if(i%params.nx==0) {
            tmp_cells[i].speeds[0] = cells[i].speeds[0];
            tmp_cells[i].speeds[1] = cells[i+params.nx-1].speeds[1];
            tmp_cells[i].speeds[2] = cells[i-params.nx].speeds[2];
            tmp_cells[i].speeds[3] = cells[i+1].speeds[3];
            tmp_cells[i].speeds[4] = cells[i+params.nx].speeds[4];
            tmp_cells[i].speeds[5] = cells[i-1].speeds[5];
            tmp_cells[i].speeds[6] = cells[i-params.nx+1].speeds[6];
            tmp_cells[i].speeds[7] = cells[i+params.nx+1].speeds[7];
            tmp_cells[i].speeds[8] = cells[i+2*params.nx-1].speeds[8];
            continue;
        }
       // MPI_Wait(&recvRequest1, &status);
       // MPI_Wait(&sendRequest2, &status);
        // Last column
        if((i+1)%params.nx==0) {
            tmp_cells[i].speeds[0] = cells[i].speeds[0];
            tmp_cells[i].speeds[1] = cells[i-1].speeds[1];
            tmp_cells[i].speeds[2] = cells[i-params.nx].speeds[2];
            tmp_cells[i].speeds[3] = cells[i-params.nx+1].speeds[3];
            tmp_cells[i].speeds[4] = cells[i+params.nx].speeds[4];
            tmp_cells[i].speeds[5] = cells[i-params.nx-1].speeds[5];
            tmp_cells[i].speeds[6] = cells[i-2*params.nx+1].speeds[6];
            tmp_cells[i].speeds[7] = cells[i+1].speeds[7];
            tmp_cells[i].speeds[8] = cells[i+params.nx-1].speeds[8];
            continue;
        }
       
    }
   free(from_up_array);
   free(from_down_array);
   free(to_up_array);
   free(to_down_array);

   from_up_array = NULL;
   from_down_array = NULL;
   to_up_array = NULL;
   to_down_array = NULL;
  return EXIT_SUCCESS;
}

int collision_rebound_av_velocity(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* av_vels, int index, int start_point, int end_point)
{
    int ii,kk;                    /* generic counters */
    float u[NSPEEDS];            /* directional velocities */
    float d_equ[NSPEEDS];        /* equilibrium densities */
    int   tot_cells = 0;         /* no. of cells used in calculation */
    float tot_u = 0.0;           /* accumulated magnitudes of velocity for each cell */

    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    

    #pragma vector aligned
    for(ii=start_point; ii<end_point; ii++) {
        /* don't consider occupied cells */
        if(!obstacles[ii]) {
            /* compute local density total */
            float local_density = 0.0;
           /* for(kk=0;kk<NSPEEDS;kk++) {
                local_density += tmp_cells[ii].speeds[kk];
            }*/
            local_density += tmp_cells[ii].speeds[0];
            local_density += tmp_cells[ii].speeds[1];
            local_density += tmp_cells[ii].speeds[2];
            local_density += tmp_cells[ii].speeds[3];
            local_density += tmp_cells[ii].speeds[4];
            local_density += tmp_cells[ii].speeds[5];
            local_density += tmp_cells[ii].speeds[6];
            local_density += tmp_cells[ii].speeds[7];
            local_density += tmp_cells[ii].speeds[8];

            /* compute x velocity component */
            float u_x = (tmp_cells[ii].speeds[1] + 
                         tmp_cells[ii].speeds[5] + 
                         tmp_cells[ii].speeds[8]
                         - (tmp_cells[ii].speeds[3] + 
                            tmp_cells[ii].speeds[6] + 
                            tmp_cells[ii].speeds[7]))
                            * (1/local_density);
            /* compute y velocity component */
            float u_y = (tmp_cells[ii].speeds[2] + 
                         tmp_cells[ii].speeds[5] + 
                         tmp_cells[ii].speeds[6]
                         - (tmp_cells[ii].speeds[4] + 
                            tmp_cells[ii].speeds[7] + 
                            tmp_cells[ii].speeds[8]))
                             * (1/local_density);
            float value = (u_x * u_x) + (u_y * u_y); 
            /* accumulate the norm of x- and y- velocity components */
            tot_u += sqrt(value);
            /* increase counter of inspected cells */
            ++tot_cells;

            /* velocity squared */ 
            float u_sq = u_x * u_x + u_y * u_y;
            cells[ii].speeds[0] = (tmp_cells[ii].speeds[0]+ params.omega * 
                                  (0.44444444 * local_density * (1.0 - u_sq * 1.5) - tmp_cells[ii].speeds[0]));
            cells[ii].speeds[1] = (tmp_cells[ii].speeds[1]+ params.omega * 
                                  (0.11111111 * local_density * (1.0 + u_x * 3.0+ (u_x * u_x) * 4.5- u_sq * 1.5)
                                   - tmp_cells[ii].speeds[1]));
            cells[ii].speeds[2] = (tmp_cells[ii].speeds[2]+ params.omega * 
                                  (0.11111111 * local_density * (1.0 + u_y * 3.0+ (u_y * u_y) * 4.5- u_sq * 1.5) 
                                   - tmp_cells[ii].speeds[2]));
            cells[ii].speeds[3] = (tmp_cells[ii].speeds[3]+ params.omega * 
                                  (0.11111111 * local_density * (1.0 - u_x * 3.0+ (u_x * u_x) * 4.5- u_sq * 1.5) 
                                   - tmp_cells[ii].speeds[3]));
            cells[ii].speeds[4] = (tmp_cells[ii].speeds[4]+ params.omega * 
                                  (0.11111111 * local_density * (1.0 - u_y * 3.0+ (u_y * u_y) * 4.5- u_sq * 1.5) 
                                   - tmp_cells[ii].speeds[4]));
            cells[ii].speeds[5] = (tmp_cells[ii].speeds[5]+ params.omega * 
                                  (0.027777777 * local_density * (1.0 + (u_x + u_y) * 3.0+ ((u_x + u_y) * (u_x + u_y)) * 4.5- u_sq * 1.5)                                   - tmp_cells[ii].speeds[5]));
            cells[ii].speeds[6] = (tmp_cells[ii].speeds[6]+ params.omega * 
                                  (0.027777777 * local_density * (1.0 + (- u_x + u_y) * 3.0+ ((- u_x + u_y)* (- u_x + u_y)) * 4.5
                                   - u_sq * 1.5) - tmp_cells[ii].speeds[6]));
            cells[ii].speeds[7] = (tmp_cells[ii].speeds[7]+ params.omega * 
                                  (0.027777777 * local_density * (1.0 + (- u_x - u_y) * 3.0+ ((- u_x - u_y) * (- u_x - u_y)) * 4.5
                                   - u_sq * 1.5) - tmp_cells[ii].speeds[7]));
            cells[ii].speeds[8] = (tmp_cells[ii].speeds[8]+ params.omega * 
                                  (0.027777777 * local_density * (1.0 + (u_x - u_y) * 3.0+ ((u_x - u_y) * (u_x - u_y)) * 4.5- u_sq * 1.5)                                   - tmp_cells[ii].speeds[8]));
           
        } else {
            /* called after propagate, so taking values from scratch space
            ** mirroring, and writing into main grid */
            cells[ii].speeds[1] = tmp_cells[ii].speeds[3];
            cells[ii].speeds[2] = tmp_cells[ii].speeds[4];
            cells[ii].speeds[3] = tmp_cells[ii].speeds[1];
            cells[ii].speeds[4] = tmp_cells[ii].speeds[2];
            cells[ii].speeds[5] = tmp_cells[ii].speeds[7];
            cells[ii].speeds[6] = tmp_cells[ii].speeds[8];
            cells[ii].speeds[7] = tmp_cells[ii].speeds[5];
            cells[ii].speeds[8] = tmp_cells[ii].speeds[6];
        }
    }

    int total_cells = 0;
    MPI_Reduce(&tot_cells, &total_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    float total_u = 0.0;
    MPI_Reduce(&tot_u, &total_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        av_vels[index] = total_u / (float)total_cells;
    }

    return EXIT_SUCCESS;
}

int initialise(const char* paramfile, const char* obstaclefile,
           t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, 
           int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE   *fp;            /* file pointer */
  int    ii,jj;          /* generic counters */
  int    xx,yy;          /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */ 
  int    retval;         /* to hold return value for checking */
  float w0,w1,w2;       /* weighting factors */

  /* open the parameter file */
  fp = fopen(paramfile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input parameter file: %s", paramfile);
    die(message,__LINE__,__FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp,"%d\n",&(params->nx));
  if(retval != 1) die ("could not read param file: nx",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->ny));
  if(retval != 1) die ("could not read param file: ny",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->maxIters));
  if(retval != 1) die ("could not read param file: maxIters",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
  if(retval != 1) die ("could not read param file: reynolds_dim",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->density));
  if(retval != 1) die ("could not read param file: density",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->accel));
  if(retval != 1) die ("could not read param file: accel",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->omega));
  if(retval != 1) die ("could not read param file: omega",__LINE__,__FILE__);

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
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
  if (*cells_ptr == NULL) 
    die("cannot allocate memory for cells",__LINE__,__FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
  if (*tmp_cells_ptr == NULL) 
    die("cannot allocate memory for tmp_cells",__LINE__,__FILE__);
  
  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int*)*(params->ny*params->nx));
  if (*obstacles_ptr == NULL) 
    die("cannot allocate column memory for obstacles",__LINE__,__FILE__);

  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      /* centre */
      (*cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */ 
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*obstacles_ptr)[ii*params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input obstacles file: %s", obstaclefile);
    die(message,__LINE__,__FILE__);
  }

  /* read-in the blocked cells list */
  while( (retval = fscanf(fp,"%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if ( retval != 3)
      die("expected 3 values per line in obstacle file",__LINE__,__FILE__);
    if ( xx<0 || xx>params->nx-1 )
      die("obstacle x-coord out of range",__LINE__,__FILE__);
    if ( yy<0 || yy>params->ny-1 )
      die("obstacle y-coord out of range",__LINE__,__FILE__);
    if ( blocked != 1 ) 
      die("obstacle blocked value should be 1",__LINE__,__FILE__);
    /* assign to array */
    (*obstacles_ptr)[yy*params->nx + xx] = blocked;
  }
  
  /* and close the file */
  fclose(fp);

  /* 
  ** allocate space to hold a record of the avarage velocities computed 
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float)*params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
         int** obstacles_ptr, float** av_vels_ptr)
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

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, float final_average_velocity)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  
  return final_average_velocity * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  int ii,jj,kk;        /* generic counters */
  float total = 0.0;  /* accumulator */

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
    total += cells[ii*params.nx + jj].speeds[kk];
      }
    }
  }
  
  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  int ii,jj,kk;                 /* generic counters */
  const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* an occupied cell */
      if(obstacles[ii*params.nx + jj]) {
    u_x = u_y = u = 0.0;
    pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
    local_density = 0.0;
    for(kk=0;kk<NSPEEDS;kk++) {
      local_density += cells[ii*params.nx + jj].speeds[kk];
    }
    /* compute x velocity component */
    u_x = (cells[ii*params.nx + jj].speeds[1] + 
           cells[ii*params.nx + jj].speeds[5] +
           cells[ii*params.nx + jj].speeds[8]
           - (cells[ii*params.nx + jj].speeds[3] + 
          cells[ii*params.nx + jj].speeds[6] + 
          cells[ii*params.nx + jj].speeds[7]))
      / local_density;
    /* compute y velocity component */
    u_y = (cells[ii*params.nx + jj].speeds[2] + 
           cells[ii*params.nx + jj].speeds[5] + 
           cells[ii*params.nx + jj].speeds[6]
           - (cells[ii*params.nx + jj].speeds[4] + 
          cells[ii*params.nx + jj].speeds[7] + 
          cells[ii*params.nx + jj].speeds[8]))
      / local_density;
    /* compute norm of velocity */
    u = sqrt((u_x * u_x) + (u_y * u_y));
    /* compute pressure */
    pressure = local_density * c_sq;
      }
      /* write to file */
      fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }
  for (ii=0;ii<params.maxIters;ii++) {
    fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n",message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
