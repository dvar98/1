# Algoritmos para maratón

## Descripción
Este repositorio contiene una colección de algoritmos útiles para competiciones de programación.

## Algoritmos

### Algoritmos de ordenamiento

### Algoritmo de Bubble Sort

``` python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

arr = [64, 34, 25, 12, 22, 11, 90]
bubbleSort(arr)
print("Sorted array is:", arr)
```

### Algoritmo de Quick Sort
``` python
def partition(arr, low, high):
    i = (low - 1)
    pivot = arr[high]
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)

arr = [10, 7, 8, 9, 1, 5]
n = len(arr)
quickSort(arr, 0, n - 1)
print("Sorted array is:", arr)
```

### Algoritmo de Merge Sort
``` python
def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        mergeSort(L)
        mergeSort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

arr = [12, 11, 13, 5, 6, 7]
mergeSort(arr)
print("Sorted array is:", arr)
```
### Algoritmo de Insertion Sort

``` python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

### Búsqueda Lineal

```python
def linear_search(arr, x):
    """
    Realiza una búsqueda lineal en la lista `arr` para encontrar el elemento `x`.
    :param arr: Lista en la que se va a buscar.
    :param x: Elemento que se desea encontrar.
    :return: Índice del elemento si se encuentra, de lo contrario -1.
    """
    for i in range(len(arr)):
        # Si el elemento en la posición i es igual a x, se devuelve el índice
        if arr[i] == x:
            return i
    # Si se recorre toda la lista y no se encuentra el elemento, se devuelve -1
    return -1

arr = [2, 3, 4, 10, 40]
x = 10
result = linear_search(arr, x)
print("Elemento encontrado en el índice:", result)
```

### Búsqueda Binaria

```python
def binary_search(arr, x):
    """
    Realiza una búsqueda binaria en la lista ordenada `arr` para encontrar el elemento `x`.
    :param arr: Lista ordenada en la que se va a buscar.
    :param x: Elemento que se desea encontrar.
    :return: Índice del elemento si se encuentra, de lo contrario -1.
    """
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2

        # Si x está presente en mid
        if arr[mid] < x:
            low = mid + 1
        # Si x está presente en mid
        elif arr[mid] > x:
            high = mid - 1
        # x está presente en mid
        else:
            return mid
    # Si el elemento no está presente en la lista
    return -1

arr = [2, 3, 4, 10, 40]
x = 10
result = binary_search(arr, x)
print("Elemento encontrado en el índice:", result)
```

### Búsqueda de Interpolación

``` python
def interpolation_search(arr, x):
    """
    Realiza una búsqueda de interpolación en la lista ordenada `arr` para encontrar el elemento `x`.
    :param arr: Lista ordenada en la que se va a buscar.
    :param x: Elemento que se desea encontrar.
    :return: Índice del elemento si se encuentra, de lo contrario -1.
    """
    low = 0
    high = len(arr) - 1

    while low <= high and x >= arr[low] and x <= arr[high]:
        # Si low es igual a high
        if low == high:
            if arr[low] == x:
                return low
            return -1

        # Fórmula de interpolación
        pos = low + ((high - low) // (arr[high] - arr[low]) * (x - arr[low]))

        # Si x está en la posición pos
        if arr[pos] == x:
            return pos

        # Si x es mayor, x está en la parte derecha
        if arr[pos] < x:
            low = pos + 1
        # Si x es menor, x está en la parte izquierda
        else:
            high = pos - 1

    return -1

arr = [10, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 33, 35, 42, 47]
x = 18
result = interpolation_search(arr, x)
print("Elemento encontrado en el índice:", result)
```