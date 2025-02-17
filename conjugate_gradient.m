function [x] = conjugate_gradient(A, b, x0, tol, max_iter)
    % Algoritmul pentru metoda gradientului conjugat.

    x = x0;
    # initializez gradientul initial
    r = b - A * x;
    # p este directia de cautare
    p = r;
    # numarul de iteratii
    iter = 0;

    # iterez pana cand se atinge toleranta sau numarul maxim de iteratii
    while norm(r) > tol && iter < max_iter
        # pasul optim
        alpha = (r' * r) / (p' * A * p);

        # actualizez solutia x
        x = x + alpha * p;

        # gradientul nou
        r_new = r - alpha * A * p;
        beta = (r_new' * r_new) / (r' * r);

        # actualizez noua directie de cautuare
        p = r_new + beta * p;

        # schmib valoarea gradientului
        r = r_new;
        iter = iter + 1;
    end
end

