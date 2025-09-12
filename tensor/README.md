# Tensor++ - Libreria C++ per tensori multi-dimensionali

## Descrizione

Tensor++ è una libreria C++ che fornisce una classe template generica per la gestione di tensori (array multi-dimensionali) di qualsiasi tipo e dimensione.
Include operatori matematici, funzioni di utilità per matrici e tuple, e supporto per operazioni avanzate come contrazione di tensori e calcolo del determinante.

## Funzionalità principali
- Classe `tensor<A, N>`: tensore generico di tipo `A` e dimensione `N`
- Alias `matrix<A>` e `tuple<A>` per matrici 2D e tuple 1D
- Operatori matematici: somma, sottrazione, moltiplicazione tra tensori e scalari
- Funzioni di utilità: trasposizione, traccia, submatrice, adjugata, determinante
- Controllo automatico degli indici e delle dimensioni

## Utilizzo

### Inclusione
```cpp
#include "tensor.h"
```

### Creazione di un tensore
```cpp
tensor<double, 3> t(2, 3, 4); // Tensore 2x3x4 di double
```

### Operazioni
```cpp
auto t2 = t + t;      // Somma tra tensori
auto t3 = t + 5.0;    // Somma con scalare
auto t4 = t * 2.0;    // Moltiplicazione con scalare
```

### Alias
```cpp
matrix<double> m(3, 3); // Matrice 3x3
tuple<int> v(5);        // Tuple di 5 elementi
```

### Funzioni di utilità
```cpp
auto mt    = tensor::T(m);   // Matrice trasposta
auto trace = tensor::tr(m);  // Traccia della matrice
auto det   = tensor::det(m); // Determinante della matrice
auto adj   = tensor::adj(m); // Matrice aggiunta

// Righe e colonne della sottomatrice
auto row   = tensor::tuple<int>(1);
auto col   = tensor::tuple<int>(2);

row(0)     = 1; // Riga 1 della matrice da rimuovere
col(0)     = 3; // Colonna 3 della matrice da rimuovere
col(1)     = 4; // Colonna 4 della matrice da rimuovere
auto subm  = tensor::submatrix(m, row, col); // Sottomatrice rispetto alle righe/colonne
```

## API Principali

### Classe `tensor<A, N>`
- Costruttori: dimensioni variabili, copia, assegnazione
- Operatori: somma, sottrazione, moltiplicazione
- Metodi statici: `matrix<A> zero(...)` (matrice nulla), `matrix<A> identity(...)` (matrice identità)

### Funzioni globali
- `matrix<A> T(matrix<A>)` — trasposizione
- `A tr(matrix<A>)` — traccia
- `matrix<A> submatrix(matrix<A>, tuple<int>, tuple<int>)` — sottomatrice
- `matrix<A> adj(matrix<A>)` — matrice aggiunta
- `A det(matrix<A>)` — determinante

## Esempio completo
```cpp
#include "tensor.h"
using namespace tensor;

int main() {
    matrix<double> m(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            m(i, j) = i + j;
    auto mt = T(m);
    double trace = tr(m);
    double detm = det(m);
}
```

## Note tecniche
- Tutte le operazioni sono template per efficienza e flessibilità
- Gli errori di dimensione e indice generano eccezioni (`throw`)
- La memoria è gestita manualmente tramite puntatori

## Autore
@LucaPrevi0o - Luca Previati

## Licenza
MIT
