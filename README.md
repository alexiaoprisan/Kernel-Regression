
# Metoda 1: Detecția anomaliilor

## 1.1 Introducere

Analiza seturilor de date folosite pentru antrenare (Training Dataset) începe cu un prim pas prin care se detectează anomaliile (outliers). O anomalie în setul de date poate fi înțeleasă ca un subset al datelor de antrenare care nu se potrivește cu întreg setul de date. Scopul acestei metode este să înțelegem ce înseamnă această (ne)potrivire.

Această metodă se numește **detecția anomaliilor** (Anomaly Detection) și este esențială pentru modelele care doresc să aibă un set de date consistent, fără erori, și să identifice anomaliile din datele de antrenare/testare.

## 1.2 Prezentarea metodei

### Setul de date folosit pentru analiză

Într-un set de date, avem mai multe elemente de forma:

```text
{x(1), x(2), ..., x(m)}
```

Fiecare element este un vector de dimensiune n, `x(i) ∈ Rn`. Pentru acest set de date, definim vectorul medie `μ ∈ Rn`, pe componente, astfel:

```text
μ(i) = 1/m Σ (x(k)(i)), i = 1..n
```

De asemenea, definim o matrice specifică numită **matricea de varianță**:

```text
Σ = (1/m) Σ (x(i) - μ)(x(i) - μ)^T, Σ ∈ Rn×n
```

Funcția de probabilitate pentru apariția unui element `x(i)` în dataset este:

```text
f(x(i)) = (1 / (2π)^(n/2) * det(Σ)^(1/2)) * exp(-(1/2) * (x(i) - μ)^T * Σ^(-1) * (x(i) - μ))
```

### Definiția unei anomalii

Un vector `x(i)` este considerat o anomalie față de setul de date dacă funcția de probabilitate `f(x(i))` este mai mică decât un anumit prag `ε`:

```text
f(x(i)) < ε
```

### Determinarea outlier-ilor (Anomaly Detection)

Determinarea outlier-ilor se face în trei etape:

1. **Calcularea mediei `μ` și a matricei de varianță `Σ`** pentru dataset-ul considerat, precum și calcularea funcției `f` pentru toate elementele din cadrul dataset-ului.
2. **Estimarea factorului `ε`** din setul de antrenament. Pentru fiecare valoare `ε`, se calculează parametrii:
   - **Precision**: procentajul outlier-ilor adevărați determinați (true positives) din totalul outlier-ilor determinați (total positives).
   - **Recall**: procentajul outlier-ilor adevărați determinați (true positives) din totalul outlier-ilor reali (true positives + false negatives).
   - **F1**: parametrul care trebuie să fie minim pentru cel mai bun `ε`:

     ```text
     F1 = (2 * precision * recall) / (precision + recall)
     ```

3. După alegerea celui mai bun `ε` astfel încât parametrul F1 să fie minim, determinăm toți outlierii din dataset-ul de testare ca fiind toți vectorii `x(k)` pentru care `f(x(k)) < εbest`.

## 1.3 Funcțiile din cadrul bibliotecii

Această metodă constă în implementarea celor trei pași menționați mai sus și includerea unor funcții care calculează media, varianța și probabilitatea pentru un dataset.

### Funcții implementate:

- **`estimate_gaussian(X)`**: Această funcție determină media și varianța pentru dataset-ul dat.
- **`gaussian_distribution(X, mean_value, variance)`**: Calculează densitatea de probabilitate pentru dataset-ul dat.
- **`check_predictions(predictions, truths)`**: Determină numărul de pozitivi adevărați (true positives), pozitivi falși (false positives) și negativi falși (false negatives).
- **`optimal_threshold(truths, probabilities)`**: Determină cel mai bun factor `ε` pe baza valorii F1, generând mai mulți `ε` pentru a verifica rezultatele.

## 2. Implementare

În cadrul implementării, setul de date a fost calculat în două funcții principale: **"estimate_gaussian"** și **"multivariate_gaussian"**, în care am determinat vectorul de medie, matricea de varianță și probabilitatea de apariție a unui element pe baza formulelor prezentate în enunț.

### Determinarea outlier-ilor

Pentru determinarea outlier-ilor, am folosit funcția **"check_predictions"**, care a returnat valorile de **false positives**, **false negatives** și **true positives** pe baza unui set de date etichetat. Aceste valori au fost folosite în funcția **"metrics"** pentru a calcula pentru fiecare valoare de `ε`, parametrii: **precision**, **recall** și **F1**. Acești parametri au fost utilizați pentru alegerea celui mai bun `ε`.

În funcția **"optimal_threshold"**, am generat mai mulți `ε` și am verificat valoarea F1 pentru fiecare, alegând cel mai bun prag.

## Concluzie

Această metodă de detecție a anomaliilor este utilă pentru identificarea datelor care nu se potrivesc cu modelele de antrenare și poate fi folosită pentru îmbunătățirea calității datelor înainte de antrenarea unui model de învățare automată. Parametrul `ε` ales corect permite identificarea anomaliilor într-un mod eficient și precis.
