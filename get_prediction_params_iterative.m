function a = get_prediction_params_iterative(K, y, lambda)
  % K - Matricea kernel
  % y - Vectorul de răspunsuri observate
  % lambda - Parametrul de regularizare

  % dimensiunea matricei K
  [m, n] = size(K);
  a = zeros(m, 1);

  # vectorul initial pentru metoda gradientului conjugat
  x0 = zeros(m, 1);
  # setez toleranta si numarul maxim de iteratii
  tol = 1e-6;

  # matricea sistemului pentru metoda gradientului conjugat
  A = K + lambda * eye(m);

  # aplic metoda gradientului conjugat
  a = conjugate_gradient(A, y, x0, tol, 1000);

endfunction

