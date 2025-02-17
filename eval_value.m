function pred = eval_value(x, X, f, f_param, a)
  # TO DO: pentru un vector de input x, preziceti valoarea acestuia in
  # in functie de toti vectorii de input folositi pentru a antrena modelul
  # folosind functia de kernel f care are ca al 3-lea parametru f_param
  # si vectorul coloana a

  # calculez kernel-ul între x si toate punctele din setul de antrenare
  K_x = zeros(size(X, 1), 1);

  for i = 1:size(X, 1)

    vector = X(i, :);
    K_x(i) = f(x, vector, f_param);
  end

  # calculez predictia
  # am folosit produsul scalar dintre vectorul de kernel al lui x si vectorul de parametri a
  K_x_transpus = K_x';
  pred = K_x_transpus * a;

  endfunction
