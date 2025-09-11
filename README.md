# Neural++
## Libreria per motori neurali MLP in C++

Neural++ è una libreria C++ per la manipolazione di reti neurali basate su MLP (Multi-Layer Perceptron), pensata per applicazioni di machine learning e data science.

Si basa sul funzionamento della classe `tensor::tensor`, all'interno della libreria `tensor.h`, che fornisce tutte le funzionlità di algebra lineare.
La classe `neural::network`, all'interno del proprio namespace, implementa funzioni per la creazione di reti neurali con dimensione variabile, il training, e anche il caricamento di dataset pre-generati tramite file CSV.

## Struttura del progetto

```
include/
  tensor.h         # Classe tensor generica multi-dimensionale, alias matrix e tuple, funzioni matematiche
  cnetwork/
    function.h     # Funzioni di attivazione, operazioni matematiche per reti neurali
    neural.h       # Strutture e algoritmi per reti neurali (layer, forward, backward, training)
    dataset/
      csv.h        # Caricamento e parsing di dataset CSV, split in features/target
```

## Funzionalità principali

### tensor.h
- **Classe template `tensor<A, N>`**: array multi-dimensionale generico
- **Alias `matrix<A>`**: matrice 2D
- **Alias `tuple<A>`**: vettore 1D
- **Operatori aritmetici**: somma, sottrazione, prodotto, scalari
- **Funzioni matematiche**: trasposizione, traccia, determinante, sottostruttura, aggiunta
- **Gestione memoria**: costruttori, copy, assegnazione, confronto

### function.h
- **Funzioni di attivazione**: sigmoid, relu, tanh, softmax
- **Derivate delle funzioni**: per backpropagation
- **Operazioni matematiche**: normalizzazione, funzioni di costo

### neural.h
- **Definizione layer**: fully connected, attivazione, output
- **Algoritmi di training**: forward, backward, aggiornamento pesi
- **Gestione rete**: creazione, configurazione, inferenza

### csv.h
- **Caricamento CSV**: parsing efficiente di file CSV
- **Estrazione labels**: lettura delle intestazioni
- **Split dataset**: separazione automatica in features e target
- **Restituzione dati**: tuple/matrix/tensor per uso diretto in reti neurali

## Esempio d'uso

```cpp
#include "include/tensor.h"
#include "include/cnetwork/dataset/csv.h"
#include "include/cnetwork/neural.h"

using namespace neural;

int main() {
    // Carica dataset
    auto data = csv::load_split("Dataset.csv", 1);
    auto features = data(0);
    auto targets = data(1);

    // Crea e addestra una rete neurale
    // ... vedi neural.h per dettagli ...
}
```

## Requisiti
- C++11 o superiore
- Nessuna dipendenza esterna

## Autori
- LucaPrevi0o

## Licenza
MIT
