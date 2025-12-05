# Klasyfikacja zbioru Wine – normalizacja i strojenie hiperparametrów

## 1. Baseline – model bez normalizacji

Jako punkt odniesienia (baseline) wykorzystałam model z poprzednich zajęć, bez warstwy normalizującej na wejściu.  
Model miał architekturę:

- Dense(32, ReLU)
- Dense(16, ReLU)
- Dense(3, Softmax)

Na podstawie pliku `baseline_results.json`:

- liczba epok: **50**
- batch size: **16**
- learning rate: **0.001**
- **val_acc_best ≈ 0.79**
- **test_acc ≈ 0.91**

Oznacza to, że model dość dobrze rozróżnia trzy klasy win, ale zdarzają się pomyłki – baseline służy dalej jako punkt 
odniesienia dla normalizacji i tuningu.

---

## 2. Modele z normalizacją

Następnie dodałam warstwę **`tf.keras.layers.Normalization`** na wejściu modelu.  
Dwa trenowane warianty:

1. **Model 1 (z normalizacją)**  
   - Normalization (adaptowana na `X_train`)  
   - Dense(32, ReLU)  
   - Dense(16, ReLU)  
   - Dense(3, Softmax)  

2. **Model 2 (z normalizacją)**  
   - Normalization  
   - Dense(64, SELU)  
   - Dropout(0.2)  
   - Dense(32, ReLU)  
   - Dense(16, Tanh)  
   - Dense(3, Softmax)  

Parametry treningu były takie same jak w baseline (żeby porównanie było uczciwe):

- Model 1: 50 epok, batch size = 16, learning rate = 0.001  
- Model 2: 70 epok, batch size = 8, learning rate = 0.0005  

Z plików `norm_results_model1.json` i `norm_results_model2.json`:

- **Model 1 (norm)**:  
  - `val_acc_best = 1.0`  
  - `test_acc = 1.0`

- **Model 2 (norm)**:  
  - `val_acc_best = 1.0`  
  - `test_acc = 1.0`

Po dodaniu normalizacji **dokładność na walidacji i teście wzrosła do 100%**, co jest ogromnym skokiem względem 
baseline’u (~0.79/0.91).

---

## 3. Strojenie hiperparametrów – Keras Tuner

Do optymalizacji wykorzystałam **Keras Tuner** (`RandomSearch`).  
Strojonym modelem był uproszczony wariant sieci z normalizacją:

- Normalization (adaptowana na `X_train`)
- Dense(`units_1`, ReLU)
- Dropout(`dropout_rate`)
- Dense(3, Softmax)

Jest to mała sieć (jedna warstwa ukryta + Dropout, < 1 tys. parametrów trenowalnych), wystarczająca dla zbioru Wine.

Strojonymi hiperparametrami były:

1. **learning_rate** – `Float[1e-4, 1e-2]` (skala logarytmiczna),
2. **units_1** – liczba neuronów w warstwie ukrytej (16–128, krok 16),
3. **dropout_rate** – współczynnik Dropout (0.0–0.5, krok 0.1).

Tuner `RandomSearch` uruchomiono z parametrami:  
`max_trials = 20`, `executions_per_trial = 1`.

Z pliku `tuner_results.json` najlepsze hiperparametry to:

- `learning_rate ≈ 0.000925`
- `units_1 = 48`
- `dropout_rate = 0.4`

Wyniki najlepszego modelu z tunera:

- **test_loss ≈ 0.1408**
- **test_acc = 1.0**

Model zapisano w pliku `wine_best_model_tuned.keras`.

Macierz pomyłek dla najlepszego modelu (plik `confusion_matrix_tuned.csv`) ma postać:

\[
\begin{bmatrix}
17 & 0 & 0 \\
0 & 8 & 0 \\
0 & 0 & 10
\end{bmatrix}
\]

czyli wszystkie 35 próbek ze zbioru testowego zostały poprawnie sklasyfikowane.

---

## 4. Uruchomienie lokalne vs zdalne (CPU vs GPU)

Kod był uruchamiany zarówno lokalnie na **CPU**, jak i na zdalnej maszynie z **GPU** (ten sam skrypt `Wina2_norm.py`).  
W obu środowiskach:

- modele z normalizacją osiągają **test_acc = 1.0**,  
- model po tuningu również ma **test_acc = 1.0**,  
- macierze pomyłek są diagonalne (brak błędnych klasyfikacji).

Różnica dotyczy głównie **czasu uczenia** – na GPU trening (szczególnie wielokrotne uruchomienia w tunerze) jest 
wyraźnie szybszy niż na CPU, natomiast jakość modeli pozostaje taka sama.

---

## 5. Podsumowanie

1. **Baseline** bez normalizacji osiąga ok. **0.79** dokładności walidacyjnej i ok. **0.91** na zbiorze testowym – 
stanowi punkt odniesienia.
2. Dodanie warstwy **Normalization** na wejściu radykalnie poprawia wyniki (obie architektury osiągają **1.0** 
dokładności na walidacji i teście).
3. Strojenie hiperparametrów (`learning_rate`, `units_1`, `dropout_rate`) metodą **RandomSearch** pozwala dobrać 
kompaktowy model z Dropoutem, który również osiąga **100%** dokładności.
4. Macierze pomyłek dla najlepszych modeli są diagonalne – klasy wina są całkowicie rozróżnione.
5. Uruchomienie na maszynie z **GPU** przyspiesza czas trenowania i tuningu, ale nie zmienia jakości końcowego modelu 
względem uruchomienia na **CPU**.

