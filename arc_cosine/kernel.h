/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*   Copyright: Thorsten Joachims                                       */
/*   Date: 16.12.97                                                     */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. Just replace the line 
             return((double)(1.0)); 
   with your own kernel. */

  /* Example: The following computes the polynomial kernel. sprod_ss
              computes the inner product between two sparse vectors. 

      return((CFLOAT)pow(kernel_parm->coef_lin*sprod_ss(a,b)
             +kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
  */

/* If you are implementing a kernel that is not based on a
   feature/value representation, you might want to make use of the
   field "userdefined" in SVECTOR. By default, this field will contain
   whatever string you put behind a # sign in the example file. So, if
   a line in your training file looks like

   -1 1:3 5:6 #abcdefg

   then the SVECTOR field "words" will contain the vector 1:3 5:6, and
   "userdefined" will contain the string "abcdefg". */

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

double custom_kernel(KERNEL_PARM *kernel_parm, SVECTOR *a, SVECTOR *b) 
     /* plug in you favorite kernel */                          
{
  //return((double)(1.0));
  double j_one, k_xx_l, k_yy_l, k_xy_l;
  double j_two, theta;
  int no_of_layers, i, n_l;
  static int count=0;

  no_of_layers = atoi(kernel_parm->custom)%10;
  if(a->twonorm_sq < 0)
	a->twonorm_sq = sprod_ss(a,a);
  if(b->twonorm_sq < 0)  
	b->twonorm_sq = sprod_ss(b,b);

  k_xy_l = sprod_ss(a,b);
  k_xx_l = a->twonorm_sq;
  k_yy_l = b->twonorm_sq;

  for (i=1; i<=no_of_layers; i++) {
    theta = acos(MAX(MIN(k_xy_l / sqrt(k_xx_l * k_yy_l), (double) 1), (double) -1));
    n_l = kernel_parm->custom[i-1] - '0';
    k_xy_l = pow(k_xx_l * k_yy_l, n_l/2) / M_PI * compute_J(n_l, theta);
    if (i < no_of_layers) {
      k_xx_l  = pow(k_xx_l, n_l) / M_PI * compute_J(n_l, 0);
      k_yy_l  = pow(k_yy_l, n_l) / M_PI * compute_J(n_l, 0);
    }
  }
  return k_xy_l;
}


double compute_J(double N, double theta)
{
    if (0 == N)
        return M_PI - theta;
    if (1 == N)
        return sin(theta) + (M_PI - theta) * cos(theta);
    if (2 == N)
        return 3*sin(theta)*cos(theta) + (M_PI - theta)*(1 + 2*pow(cos(theta), 2));
    if (3 == N)
        return 4*pow(sin(theta), 3) + 15*sin(theta)*pow(cos(theta), 2) + (M_PI - theta)*(9*pow(sin(theta),2)*cos(theta) + 15*pow(cos(theta),3));
    return 0;
}
