Wprowadzenie 

W zadaniu wykorzystano zbiór Wine z repozytorium UCI. Dane wczytano z pliku wine.data (pandas → numpy), potasowano i podzielono na zbiór treningowy (80%) oraz testowy (20%). Zmienną zależną (klasa 1/2/3) zakodowano w postaci one-hot, co pozwoliło zastosować na wyjściu warstwę softmax oraz funkcję celu categorical cross-entropy.

Krótki opis modeli:

Zaimplementowano dwa modele typu Sequential:

Model 1 (prosty MLP)

Wejście: 13 cech.

Warstwa ukryta 1: 32 neurony, aktywacja ReLU (hidden_1).

Warstwa ukryta 2: 16 neuronów, aktywacja ReLU (hidden_2).

Warstwa wyjściowa: 3 neurony, aktywacja softmax (output).

Trening: 50 epok, batch_size=16, learning_rate=0.001, optymalizator Adam.

Model 2 (głębszy, bardziej złożony)

Warstwa 1: 64 neurony, aktywacja SELU, inicjalizacja HeNormal (hidden_1_selu).

Dropout 0.2 (dropout_1).

Warstwa 2: 32 neurony, aktywacja ReLU (hidden_2_relu).

Warstwa 3: 16 neuronów, aktywacja tanh (hidden_3_tanh).

Wyjście: 3 neurony, softmax.

Trening: 70 epok, batch_size=8, learning_rate=0.0005, optymalizator Adam.



Krzywe uczenia i dokładność modeli

Krzywe uczenia zarejestrowano w TensorBoard (wykres epoch_accuracy). Na załączonym wykresie widoczne są cztery przebiegi: accuracy dla części treningowej i walidacyjnej dla obu modeli.

Dla Modelu 1 dokładność na zbiorze treningowym rośnie z ok. 0,4 do ok. 0,8, a dokładność walidacyjna stabilizuje się w okolicach 0,75–0,78. Krzywe trening/validacja są zbliżone, co sugeruje dobrą zbieżność i brak silnego przeuczenia.

W przypadku Modelu 2 accuracy pozostaje znacznie niższa – ok. 0,46 na zbiorze treningowym i ok. 0,38 na walidacyjnym, a przebieg jest niemal płaski, co oznacza, że model nie uczy się efektywnie.

Na zbiorze testowym Model 1 osiągnął dokładność rzędu ~0,77, natomiast Model 2 ok. 0,45–0,5 (tu możesz wpisać swoje dokładne wartości z logów).

Wnioski

Lepszy okazał się Model 1 – prostszy, z mniejszą liczbą parametrów. Przy tak małym zbiorze (178 obserwacji) bardziej złożona architektura Modelu 2 (więcej warstw, różne aktywacje, Dropout) oraz mocniejsza regularyzacja mogły utrudnić uczenie i doprowadzić do zbyt słabego dopasowania. W efekcie Model 1 generalizuje lepiej i to właśnie jego wykorzystano jako model „produkcyjny” w skrypcie z argparse, który zwraca klasę wina (1/2/3) na podstawie 13 podanych cech.