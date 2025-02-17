function retval = polynomial_kernel (x, y, d)
  # TO DO: implement polynomial kernel function

  x = x';
  y = y';
  retval = (1 + y' * x).^d;

  endfunction
