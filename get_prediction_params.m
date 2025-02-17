function [a] = get_prediction_params (K, y, lambda)
  # TO DO: folosind metode neiterative, implementati logica
  # pentru a obtine vectorul coloana a, asa cum este descris in enuntul temei

  # n este dimensiunea matricei de kernel
  n = size(K, 1);

  # adaug regularizarea la matricea de kernel
  K_reg = K + lambda * eye(n);

  # am factorizat matricea de kernel cu algoritmul Cholesky
  L = chol(K_reg, 'lower');

  # rezolv sistemul liniar L * z = y
  z = L \ y;

  # rezolv sistemul liniar L' * a = z
  a = L' \ z;
  endfunction
