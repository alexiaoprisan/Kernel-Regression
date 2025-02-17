function [L] = cholesky (A)
  # TO DO: realizati descompunerea cholesky pentru o matrice
  # positiv definita

  [n n] = size(A);
  L = zeros(n);

  for i = 1 : n
    for j = 1 : i-1

      s = 0;
      for k = 1 : j-1
        s = s+L(i,k)*L(j,k);
      endfor
      L(i,j) = (A(i,j)-s)/L(j,j);

    endfor

    s = 0;
    for k = 1 : i-1
      s = s + L(i,k)*L(i,k);
    endfor

    L(i,i) = sqrt(A(i,i)-s);
  endfor

  endfunction
