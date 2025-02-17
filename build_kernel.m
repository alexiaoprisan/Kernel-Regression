function [K] = build_kernel (X, f, f_param)
  # Construiti matricea K (matricea kernel-urilor) asa cum este
  # descrisa in enuntul temei pornind de la datele de intrare X
  # Functia de kernel este descrisa de parametrul f si foloseste f_param
  # ca al 3-lea parametru

  # calculez numarul de exemple din datele de intrare
  n = size(X, 1);

  # initializez matricea kernel-urilor
  K = zeros(n, n);

  # construiesc matricea
  for i = 1 : n
    x_i = X(i, :);
    for j = 1 : n
      x_j = X(j, :);
      # calculez valoarea kernel-ului intre exemple
      K(i, j) = f(x_i, x_j, f_param);
    end
  end

  endfunction
