
## 2 Metoda 2: Kernel Regression

### 2.1 Introducere
În viața de zi cu zi ne întâlnim cu ideea de predicție (estimare) pentru diverse activități sau fenomene. De exemplu, ne dorim să estimăm cât de mult timp ne va consuma o temă sau cât de mult timp vom sta în trafic, dacă va ploua sau nu la final de săptămână, etc. Evident că răspunsurile la astfel de întrebări depind de foarte mulți parametri și ne este destul de greu să găsim relații de cauzalitate între ei.

Astfel, pentru seturi de date foarte mari și pentru generarea unor predicții cât mai precise avem nevoie de diferite modele de Învățare Automată (en: Machine Learning, ML). 

Pentru a doua sarcină de lucru propusă veți avea de implementat metodele numerice asociate unui algoritm de Supervised Machine Learning denumit Kernel Regression. În semestrul I, la ALGAED, cu siguranță ați discutat despre dreapta de regresie (ca în exemplul grafic din Figura 1) ca dreapta ce stabilește cea mai potrivită dependență liniară.

Figura 1: Dreapta de regresie ce trece printr-un set de date (en: Linear Regression).

Totuși, majoritatea fenomenelor pe care le avem de prelucrat nu sunt neapărat liniare. În multe cazuri încercăm să ne ”liniarizăm” datele pentru a reuși să aplicăm diferite metode numerice matriceale pe acestea.

### 2.2 Cadru teoretic & Explicații
Regresia reprezintă, în esență, găsirea unor legături între anumite date. Aceasta se reduce la minimizarea unei funcții de cost și a pierderilor asociate (vom explica mai detaliat în paragrafele următoare aceste concepte).

Să presupunem că avem mai multe perechi de forma (x(i), y(i)), unde x(i) reprezintă datele de intrare cărora li se asociază valoarea de ieșire y(i), cu x(i) ∈ Rk și y(i) ∈ R. Am menționat că pentru acest tip de regresie avem nevoie de o funcție care să reprezinte această liniarizare a datelor ϕ : Rk → Rn.

De exemplu, dacă avem un set de puncte 2D pe o parabolă ce trece prin origine, fiind simetrică față de axa Oy, o astfel de funcție ar putea fi ϕ(x) = x² și astfel funcția y(x²) ar putea reprezenta o dependență liniară între seturile de puncte y(i) și x(i) (evident că în acest exemplu s-a considerat k = 1 și n = 1).

Obiectivul nostru este determinarea coeficienților specifici (în engleză se numesc weights) care să estimeze cât mai bine ieșirea (funcția y) pentru un set de date de intrare dat. Din punct de vedere numeric, aceasta s-ar traduce în determinarea unui vector θ ∈ Rn din care să rezulte o predicție de forma ypred = θT ϕ(x).

Asemenea dreptei de regresie care minimizează pătratul erorii dintre valoarea reală și predicție, și noi vom alege o funcție de cost similară:

```
Jcost(θ) = Σm i=1 (y(i) − θT ϕ(x(i)))² + λ ∗ ∥θ∥₂²,
```

unde:
- ∥θ∥₂ reprezintă norma 2 (euclideană) a coeficienților modelului, adică ∥θ∥₂² = Σn i=1 |θi|².
- λ ∈ R⁺ este parametrul care controlează regularizarea (în metodele de ML acest parametru este important pentru o evaluare cât mai bună a modelului și pentru a nu apărea anomalii datorate bias-ului setului de date. În soluția propusă de voi trebuie să țineți cont de el în funcțiile pe care le aveți de implementat, dar noi vă vom oferi acest parametru în teste, nu îl aveți de evaluat/determinat).

Vom prelucra această funcție de cost, astfel încât să obținem minimul dorit. Vom începe prelucrarea prin a pune condiția necesară asupra gradientului funcției (echivalentul derivatei unei funcții) să fie nul:

```
dJ/dθ = −2 Σm i=1 [(y(i) − θT ϕ(x(i)))ϕ(x(i))] + 2λθ = 0.
```

Vom folosi următoarea notație, αi = y(i) − θT ϕ(x(i)), pentru a exprima vectorul θ optim:

```
θ = (1/λ) Σm i=1 αiϕ(x(i)).
```

Ca observație, θ optim trebuie să fie o combinație liniară a vectorilor de intrare (input) cărora li se aplică funcția de "liniarizare" datorată regularizării pe care am adăugat-o. Scopul nostru acum este determinarea parametrilor αi (i = 1, ...,m):

```
αi = y(i) − θT ϕ(x(i)) = y(i) − (1/λ) Σm j=1 αjϕ(x(j))T ϕ(x(i)).
```

Astfel, ajungem la concluzia că αi depinde într-un mod liniar doar de produsele scalare dintre datele de intrare cărora li se aplică o funcție ϕ și răspunsurile inițiale.

Considerăm notația K(x(i), x(j)) = ϕ(x(j))T ϕ(x(i)) și vom denumi K : Rk × Rk → R funcție de kernel. Această funcție reprezintă un produs scalar peste Rk.

### 2.3 Tipuri de kernel
Rolul acestor kernel-uri este de a ne oferi o modalitate de a estima parametri necesari în funcție de gradul maxim (din punct de vedere polinomial) pe care îl atribuim funcției ϕ. Astfel, în cadrul acestui task vom defini următoarele tipuri de kernel, a căror implementare o veți avea de realizat:

- **Linear Kernel**: K(x, y) = yTx;
- **Polynomial Kernel**: K(x, y) = (1+yTx)ᵈ, unde d este dimensiunea maximă a funcției polinomiale pe care o avem de aproximat;
- **Gaussian/Radial-Basis Kernel**: K(x, y) = exp(−∥x−y∥₂² / σ²).

### 2.4 Metoda Gradientului Conjugat
În practică, există metode iterative deterministe pentru rezolvarea ecuațiilor liniare de forma Ax = b, unde A este o matrice simetrică pozitiv semi-definită. O astfel de metodă numerică este metoda gradientului conjugat care face uz de construcția unor vectori A-ortogonali care se află în Span < b, Ab, A²b, ..., A(m−1)b > (unde m este dimensiunea spațiului vectorial) și de determinarea exactă a soluției în m pași iterativi. Astfel, inversarea unei astfel de matrici poate fi mai rapidă.

Reamintim algoritmul gradientului conjugat:

```
Conjugate Gradient Method
1: procedure conjugate_gradient(A, b, x0, tol, max_iter)
2: r(0) ← b − Ax0
3: v(1) ← r(0)
4: x ← x0
5: tolsquared ← tol²
6: k ← 1
7: while k <= max_iter and r(k-1)T r(k-1) > tolsquared do
8: tk ← r(k-1)T r(k - 1) / vkTAvk
9: x(k) ← x(k-1) + tkv(k)
10: r(k) ← r(k-1) − tkAv(k)
11: sk ← r(k)T r(k) / r(k-1)T r(k-1)
12: v(k+1) ← r(k) + skv(k)
13: k ← k + 1
14: return x
```

### 2.5 Funcțiile din cadrul bibliotecii
Pentru implementarea metodei prezentate vă propunem următorul set de funcții. Primele 3 funcții sunt reprezentate prin funcțiile de kernel, așa cum sunt explicate mai sus. Pentru simplitate acestea primesc 3 parametri (la kernelul liniar puteți ignora al 3-lea parametru în logica funcției). Doar pentru aceste 3 funcții aveți în vedere primii doi parametri că fiind vectori linie n-dimensionali.

```matlab
function retval = linear_kernel(x, y, other)
    % Aceasta este utilizata pentru implementarea functiei pentru kernelul liniar.
end

function
```

