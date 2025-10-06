# Neural++ - Libreria per motori neurali MLP in C++
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Neural++ è una libreria C++ per la manipolazione di reti neurali basate su MLP (Multi-Layer Perceptron), pensata per applicazioni di machine learning e data science.

Si basa sul funzionamento della classe `tensor::tensor`, all'interno della libreria `tensor.h`, che fornisce tutte le funzionlità di algebra lineare.
La classe `neural::network`, all'interno del proprio namespace, implementa funzioni per la creazione di reti neurali con dimensione variabile, il training, e anche il caricamento di dataset pre-generati tramite file CSV.

## Struttura del progetto

```
tensor/
    tensor.h    # Classe tensor generica multi-dimensionale, alias matrix e tuple, funzioni matematiche
cnetwork/
    function.h  # Funzioni di attivazione, operazioni matematiche per reti neurali
    neural.h    # Strutture e algoritmi per reti neurali (layer, forward, backward, training)
    dataset/
        csv.h   # Caricamento e parsing di dataset CSV, split in features/target
```

## Funzionalità principali

### [tensor.h](/tensor/tensor.h) (Libreria esterna)
- **Classe template `tensor<A, N>`**: array multi-dimensionale generico
- **Alias `matrix<A>`**: matrice 2D
- **Alias `tuple<A>`**: vettore 1D
- **Operatori aritmetici**: somma, sottrazione, prodotto, scalari
- **Funzioni matematiche**: trasposizione, traccia, determinante, sottostruttura, aggiunta
- **Gestione memoria**: costruttori, copy, assegnazione, confronto

### [function.h](/cnetwork/function.h)
- **Funzioni di attivazione**: sigmoid, relu, tanh, softmax
- **Derivate delle funzioni**: per backpropagation
- **Operazioni matematiche**: normalizzazione, funzioni di costo

### [neural.h](/cnetwork/neural.h)
- **Definizione layer**: fully connected, attivazione, output
- **Algoritmi di training**: forward, backward, aggiornamento pesi
- **Gestione rete**: creazione, configurazione, inferenza

### [csv.h](/cnetwork/dataset/csv.h)
- **Caricamento CSV**: parsing efficiente di file CSV
- **Estrazione labels**: lettura delle intestazioni
- **Split dataset**: separazione automatica in features e target
- **Restituzione dati**: tuple/matrix/tensor per uso diretto in reti neurali

## Esempio d'uso

```cpp
#include "include/tensor/tensor.h"
#include "include/cnetwork/neural.h"
#include "include/cnetwork/dataset/csv.h"
#include <stdio.h>
#include <time.h>

using namespace neural;
using namespace tensor;

int main() {

    srand(static_cast<unsigned int>(time(0)));

    try {

        printf("=== Example usage: CSV Data Loading ===\n");
        
        auto result = csv::load_split("Dataset.csv", 1);
        auto features = result.features; // Features from the dataset
        auto targets = result.targets;   // Target features

        auto batch_size = 64;          // Training/validation batch size
        auto train_samples = 2048;     // Number of samples for training
        auto validation_samples = 256; // Number of samples for validation

        // Construct training/validation matrices
        auto input_features      = matrix<float>(features.size(0), train_samples);
        auto validation_features = matrix<float>(features.size(0), validation_samples);

        for (int i = 0; i < input_features.size(0); i++) 
            for (int j = 0; j < input_features.size(1); j++) {
                input_features(i, j) = features(i, j);
            }

        for (int i = 0; i < validation_features.size(0); i++) 
            for (int j = 0; j < validation_features.size(1); j++) {
                validation_features(i, j) = features(i, j + train_samples);
            }

        // Start model training
        if (input_features.size(0) > 0 && input_features.size(1) > 0) {

            activation *sigmoid = new SIGMOID(); // activation function
            network csv_net(batch_size,                          // batch size = number of samples
                network::shape(input_features.size(0), sigmoid), // input size = number of features
                network::shape(10,                     sigmoid), // hidden layer with 10 neurons
                network::shape(8,                      sigmoid), // hidden layer with 8 neurons
                network::shape(targets.size(0),        sigmoid)  // output size = number of targets
            );

            // Training and validation for the model
            auto csv_trained = train(csv_net, input_features, targets, new MSE(), 35000, 0.001f, 0.001f);
            auto csv_output  = validate(csv_trained, validation_features, targets);

            printf("CSV Output error:\n");
            for (int k = 0; k < csv_output.size(2); k++)
                for (int i = 0; i < csv_output.size(1); i++) {

                    printf("[ ");
                    for (int j = 0; j < csv_output.size(0); j++) printf("%c%.3f ", csv_output(j, i, k) >= 0 ? ' ' : 0, csv_output(j, i, k));
                    printf("]\n");
                }
            printf("CSV training completed!\n");
            delete sigmoid; // clean memory
        } else printf("No data loaded from CSV. Please ensure 'Dataset.csv' exists and is properly formatted.\n");
        return 0;

    } catch (const char* e) {

        printf("Error: %s\n", e);
        return 1;
    }
}
```

## Requisiti
- C++11 o superiore
- Librerie esterne:
    - [`tensor.h`](#tensorh-libreria-esterna)

## Autori
- @[LucaPrevi0o](http://github.com/LucaPrevi0o) - Luca Previati

## Licenza
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
Any modification to this software is permitted.
* **Any commercial use of this software is *prohibited***. This software is intended to be used and shared without payment.
* Any re-distribution of this software is allowed, **with attribution** of the original source.
* Any re-distribution of this software *must* be provided with **share-alike license**.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
