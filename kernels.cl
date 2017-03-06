#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

void reduce(                                          
   local  double*,    
   global double*,
   int nx,
   int ny,
   int time);
   
kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            double density, double accel)
{
  /* compute weighting factors */
  double w1 = density * accel / 9.0;
  double w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);
  /**************************local memory********************/
  t_speed buff_tmp = tmp_cells[ii * nx + jj];
  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
           /* ** scratch space grid */
      buff_tmp.speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
      buff_tmp.speeds[1] = cells[ii * nx + x_w].speeds[1]; /* east */
      buff_tmp.speeds[2] = cells[y_s * nx + jj].speeds[2]; /* north */
      buff_tmp.speeds[3] = cells[ii * nx + x_e].speeds[3]; /* west */
      buff_tmp.speeds[4]  = cells[y_n * nx + jj].speeds[4]; /* south */
      buff_tmp.speeds[5] = cells[y_s * nx + x_w].speeds[5]; /* north-east */
      buff_tmp.speeds[6] = cells[y_s * nx + x_e].speeds[6]; /* north-west */
      buff_tmp.speeds[7] = cells[y_n * nx + x_e].speeds[7]; /* south-west */
      buff_tmp.speeds[8] = cells[y_n * nx + x_w].speeds[8]; /* south-east */
      tmp_cells[ii * nx + jj] = buff_tmp;

}


kernel void rebound_collision_av(global t_speed* cells,
			         global t_speed* tmp_cells,
			         global int* obstacles,
                                 global double* av_vels,
				 int nx, int ny,
				 double omega,
				 local  double* local_sums,                               
				 int time)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);
  unsigned int local_index = get_local_id(0);
  unsigned int local_index1 = get_local_id(1);
  //printf("local_id1 is %d, local_id2 is %d\n", local_index,local_index1);
  const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const double w0 = 4.0 / 9.0;  /* weighting factor */
  const double w1 = 1.0 / 9.0;  /* weighting factor */
  const double w2 = 1.0 / 36.0; /* weighting factor */
  const double res1 = 2.0 / 9.0;
  const double res2 =2.0 / 3.0;
 // unsigned int tot_cell = 0;  /* no. of cells used in calculation */
  double tot_u = 0.0;
  unsigned int index = ii * nx + jj;
  t_speed buff_cell = cells[index];
  t_speed buff_tmp = tmp_cells[index]; 
  int obs = obstacles[index];
  double value = omega; 
  /*   rebound function    */
 // if (obstacles[index])
  if (obs)
  {
  /* called after propagate, so taking values from scratch space
   ** mirroring, and writing into main grid */
      buff_cell.speeds[1] = buff_tmp.speeds[3];
      buff_cell.speeds[2] = buff_tmp.speeds[4];
      buff_cell.speeds[3] = buff_tmp.speeds[1];
      buff_cell.speeds[4] = buff_tmp.speeds[2];
      buff_cell.speeds[5] = buff_tmp.speeds[7];
      buff_cell.speeds[6] = buff_tmp.speeds[8];
      buff_cell.speeds[7] = buff_tmp.speeds[5];
      buff_cell.speeds[8] = buff_tmp.speeds[6];  
 
   }
   
   /*     collision function     */
   else
   {
	   /* compute local density total */
        double local_density = 0.0;
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        local_density += buff_tmp.speeds[0];
        local_density += buff_tmp.speeds[1];
        local_density += buff_tmp.speeds[2];
        local_density += buff_tmp.speeds[3];
        local_density += buff_tmp.speeds[4];
        local_density += buff_tmp.speeds[5];
        local_density += buff_tmp.speeds[6];
        local_density += buff_tmp.speeds[7];
        local_density += buff_tmp.speeds[8];

        /* compute x velocity component */
        double u0 = (buff_tmp.speeds[1]
                   + buff_tmp.speeds[5]
                   + buff_tmp.speeds[8]
                   - (buff_tmp.speeds[3]
                   + buff_tmp.speeds[6]
                   + buff_tmp.speeds[7]));

        double u_x = u0 / local_density;
        /* compute y velocity component */
        double u1 = (buff_tmp.speeds[2]
                   + buff_tmp.speeds[5]
                   + buff_tmp.speeds[6]
                    - (buff_tmp.speeds[4]
                   + buff_tmp.speeds[7]
                   + buff_tmp.speeds[8]));

        double u_y = u1 /local_density ;


        /* velocity squared */
        double u_sq = u_x * u_x + u_y * u_y;
       // printf("value1 is %lf,value2 is %lf\n",u0,u1);
        tot_u += sqrt(u_sq);
       
        /* directional velocity components */
        double u[NSPEEDS];
        u[1] = u_x * u_x;
        u[2] = u_y * u_y;
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u[5];  /* south-west */
        u[8] =  -u[6];  /* south-east */
       
        /* equilibrium densities */
        double d_equ[NSPEEDS];
        double t0 = w0 * local_density;
        double t1 = w1 * local_density;
        double t2 = w2 * local_density;
        d_equ[0] = t0 * (1.0 - u_sq / res2);
        /* axis speeds: weight w1 */
        d_equ[1] = t1 * (1.0 + u_x / c_sq
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
                    - u_sq / res2);
        /* diagonal speeds: weight w2 */
        d_equ[5] = t2 * (1.0 + u[5] / c_sq
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
                    - u_sq / res2);

        /* relaxation step */
        buff_cell.speeds[0] = buff_tmp.speeds[0] + value
                                * (d_equ[0] - buff_tmp.speeds[0]);
                                 
        buff_cell.speeds[1] = buff_tmp.speeds[1] + value
                                 * (d_equ[1] - buff_tmp.speeds[1]);
                                 
        buff_cell.speeds[2] = buff_tmp.speeds[2] + value
                                 * (d_equ[2] - buff_tmp.speeds[2]);

        buff_cell.speeds[3] = buff_tmp.speeds[3] + value
                                 * (d_equ[3] - buff_tmp.speeds[3]);                                                                                     
        buff_cell.speeds[4] = buff_tmp.speeds[4] + value      
                                 * (d_equ[4] - buff_tmp.speeds[4]);

        buff_cell.speeds[5] = buff_tmp.speeds[5] + value
                                * (d_equ[5] - buff_tmp.speeds[5]);

        buff_cell.speeds[6] = buff_tmp.speeds[6] + value
                                 * (d_equ[6] - buff_tmp.speeds[6]);
                                 
        buff_cell.speeds[7] = buff_tmp.speeds[7] + value
                                 * (d_equ[7] - buff_tmp.speeds[7]);
                                 
        buff_cell.speeds[8] = buff_tmp.speeds[8] + value
                                 * (d_equ[8] - buff_tmp.speeds[8]);					
   }

   cells[index] = buff_cell;
   local_sums[local_index1*8+local_index] = tot_u; 
   barrier(CLK_LOCAL_MEM_FENCE);
    reduce(local_sums,av_vels,nx,ny,time); 
  
}
void reduce(                                          
   local  double*    local_sums,
   global double*    av_vels,
   int nx,
   int ny,
   int time)                        
{                                                          
   int num_wrk_items  = get_local_size(0)*get_local_size(1);                
   int local_id1      = get_local_id(0);
   int local_id2      = get_local_id(1);             
   int group_id1      = get_group_id(0);
   int group_id2      = get_group_id(1);                   
   double sum ;
                                
   int i;                                     
    
  if (local_id2 == 0 && local_id1 == 0) {                      
      sum = 0.0;
                                 
      for (i=0; i<num_wrk_items;i++){
          sum += local_sums[i];
          
      }                                     
   
      av_vels[(group_id2*(nx/8)+group_id1)+(time*((nx/8)*(ny/8)))] = sum;
            
   }
     
}
