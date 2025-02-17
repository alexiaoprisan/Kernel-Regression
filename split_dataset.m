  function [X_train, y_train, X_pred, y_pred] = split_dataset (X, y, percentage)
    # TO DO: împarte setul de date în 2 seturi:
    # un set de training si un set de test,
    # ambele reprezentate printr-o matrice de features un vector de clase
    # percentage este un parametru considerat intre 0 si 1

    # Fiecare linie a matricii X reprezinta x^{(i)} si fiecare element de pe
    # linia coloanei y reprezinta y^{(i)}

    # numarul de date
    num_total_date = size(X, 1);

    #numarul de date pentru antrenare si testare
    num_date_antrenare = round(percentage * num_total_date);

    # Setul de antrenare
    X_train = X(1:num_date_antrenare, :);
    y_train = y(1:num_date_antrenare, :);

    # Setul de testare
    X_pred = X(num_date_antrenare+1:end, :);
    y_pred = y(num_date_antrenare+1:end, :);


  endfunction
