function [P] = get_lower_inverse (L)
  # TO DO: Determinati printr-o metoda matriceala neiterativa inversa
  # unei matrici de tipul lower

  n = size(L, 1);
  P = zeros(n);

  # iterez prin fiecare coloana
  for j = 1:n
    # setez elementele sub diagonala principala la zero
    P(j, j) = 1 / L(j, j);

    for i = (j + 1):n
      # suma pentru elementele din partea de jos a matricei
      suma = 0;
      for k = 1:(i - 1)
        suma = suma + L(i, k) * P(k, j);
      end
      # aflu elementul P(i, j)
      P(i, j) = -suma / L(i, i);
    end
  end

  endfunction
