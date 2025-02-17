function retval = gaussian_kernel (x, y, sigma)
  # TO DO: implement gaussian kernel function

  x = x';
  y = y';

  # calculez distanta Euclidiana intre x și y
  distance = norm(x - y);

  # folosesc formula pentru kernel
  retval = exp(-distance^2 / (2 * sigma^2));

  endfunction
