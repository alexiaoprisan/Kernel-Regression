function retval = linear_kernel (x, y, other)
  # TO DO: implement linear kernel function
  # Ignorati parametrul other pentru aceasta functie

  x = x';
  y = y';
  retval = y' * x;

  endfunction
