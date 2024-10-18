# Algoritmos para maratón

## Descripción
Este repositorio contiene una colección de algoritmos útiles para competiciones de programación.

## Algoritmos de ordenamiento

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

## Algoritmo de busqueda

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

## Estrucuturas de Datos

### Pila (Stack)

```python
class Stack:
    def __init__(self):
        """
        Inicializa una pila vacía.
        """
        self.items = []

    def is_empty(self):
        """
        Verifica si la pila está vacía.
        :return: True si la pila está vacía, False en caso contrario.
        """
        return len(self.items) == 0

    def push(self, item):
        """
        Añade un elemento a la pila.
        :param item: Elemento a añadir.
        """
        self.items.append(item)

    def pop(self):
        """
        Elimina y devuelve el elemento en la cima de la pila.
        :return: Elemento en la cima de la pila.
        """
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("pop from empty stack")

    def peek(self):
        """
        Devuelve el elemento en la cima de la pila sin eliminarlo.
        :return: Elemento en la cima de la pila.
        """
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError("peek from empty stack")

    def size(self):
        """
        Devuelve el número de elementos en la pila.
        :return: Número de elementos en la pila.
        """
        return len(self.items)

# Ejemplo de uso
pila = Stack()
pila.push(1)
pila.push(2)
print("Elemento en la cima:", pila.peek())
print("Elemento eliminado:", pila.pop())
print("Tamaño de la pila:", pila.size())
```

### Cola (Queue)

```python
class Queue:
    def __init__(self):
        """
        Inicializa una cola vacía.
        """
        self.items = []

    def is_empty(self):
        """
        Verifica si la cola está vacía.
        :return: True si la cola está vacía, False en caso contrario.
        """
        return len(self.items) == 0

    def enqueue(self, item):
        """
        Añade un elemento al final de la cola.
        :param item: Elemento a añadir.
        """
        self.items.insert(0, item)

    def dequeue(self):
        """
        Elimina y devuelve el elemento al frente de la cola.
        :return: Elemento al frente de la cola.
        """
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("dequeue from empty queue")

    def size(self):
        """
        Devuelve el número de elementos en la cola.
        :return: Número de elementos en la cola.
        """
        return len(self.items)

# Ejemplo de uso
cola = Queue()
cola.enqueue(1)
cola.enqueue(2)
print("Elemento al frente:", cola.dequeue())
print("Tamaño de la cola:", cola.size())
```

### Lista Enlazada (Linked List)

```python
class Node:
    def __init__(self, data):
        """
        Inicializa un nodo.
        :param data: Datos que contiene el nodo.
        """
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        """
        Inicializa una lista enlazada vacía.
        """
        self.head = None

    def is_empty(self):
        """
        Verifica si la lista enlazada está vacía.
        :return: True si la lista está vacía, False en caso contrario.
        """
        return self.head is None

    def append(self, data):
        """
        Añade un nodo al final de la lista enlazada.
        :param data: Datos que contiene el nodo.
        """
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def display(self):
        """
        Muestra los elementos de la lista enlazada.
        """
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Ejemplo de uso
lista = LinkedList()
lista.append(1)
lista.append(2)
lista.append(3)
lista.display()
```

### Árbol Binario de Búsqueda (Binary Search Tree)

```python
class TreeNode:
    def __init__(self, key):
        """
        Inicializa un nodo del árbol.
        :param key: Clave del nodo.
        """
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        """
        Inicializa un árbol binario de búsqueda vacío.
        """
        self.root = None

    def insert(self, key):
        """
        Inserta una clave en el árbol binario de búsqueda.
        :param key: Clave a insertar.
        """
        if self.root is None:
            self.root = TreeNode(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        """
        Función auxiliar para insertar una clave en el árbol.
        :param node: Nodo actual.
        :param key: Clave a insertar.
        """
        if key < node.key:
            if node.left is None:
                node.left = TreeNode(key)
            else:
                self._insert(node.left, key)
        else:
            if node.right is None:
                node.right = TreeNode(key)
            else:
                self._insert(node.right, key)

    def inorder(self):
        """
        Realiza un recorrido en orden del árbol y muestra las claves.
        """
        self._inorder(self.root)

    def _inorder(self, node):
        """
        Función auxiliar para el recorrido en orden.
        :param node: Nodo actual.
        """
        if node:
            self._inorder(node.left)
            print(node.key, end=" ")
            self._inorder(node.right)

# Ejemplo de uso
bst = BinarySearchTree()
bst.insert(10)
bst.insert(5)
bst.insert(15)
bst.insert(3)
bst.insert(7)
bst.inorder()
```
## Problemas resueltos


**Problema de las N Reinas:**

Colocar N reinas en un tablero de ajedrez de NxN de tal manera que ninguna reina pueda atacar a otra. Esto significa que no puede haber dos reinas en la misma fila, columna o diagonal.

**Objetivo:**
Encontrar todas las configuraciones posibles para colocar N reinas en el tablero, o al menos una configuración válida si existe.

**Restricciones:**
- Cada reina puede moverse horizontalmente, verticalmente y diagonalmente en cualquier dirección.
- Dos reinas no pueden compartir la misma fila, columna o diagonal.

**Ejemplo:**
Para N = 8 (Problema de las 8 Reinas), el objetivo es colocar 8 reinas en un tablero de 8x8 de manera que ninguna de ellas se ataque entre sí.

### Solución del Problema de las 8 Reinas
```python
def is_safe(board, row, col):
    """
    Verifica si es seguro colocar una reina en la posición (row, col) en el tablero.
    :param board: Tablero de ajedrez.
    :param row: Fila actual.
    :param col: Columna actual.
    :return: True si es seguro colocar la reina, de lo contrario False.
    """
    # Verifica la fila a la izquierda
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Verifica la diagonal superior a la izquierda
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[row][j] == 1:
            return False

    # Verifica la diagonal inferior a la izquierda
    for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
        if board[row][j] == 1:
            return False

    return True

def solve_n_queens_util(board, col):
    """
    Utilidad para resolver el problema de las n reinas utilizando backtracking.
    :param board: Tablero de ajedrez.
    :param col: Columna actual.
    :return: True si se puede resolver el problema, de lo contrario False.
    """
    # Si todas las reinas están colocadas, retorna True
    if col >= len(board):
        return True

    # Considera esta columna y trata de colocar una reina en cada fila
    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 1
            # Recursivamente coloca el resto de las reinas
            if solve_n_queens_util(board, col + 1):
                return True
            # Si colocar la reina en board[i][col] no lleva a una solución, la elimina
            board[i][col] = 0

    return False

def solve_n_queens(n):
    """
    Resuelve el problema de las n reinas.
    :param n: Número de reinas.
    :return: Tablero con la solución.
    """
    board = [[0 for _ in range(n)] for _ in range(n)]

    if not solve_n_queens_util(board, 0):
        print("No existe solución")
        return None

    return board

# Ejemplo de uso
n = 8
solucion = solve_n_queens(n)
if solucion:
    for fila in solucion:
        print(fila)
```

## Problema de las Buenas Permutaciones

### Descripción

Tienes una secuencia de longitud \( N \), creada eliminando algunos elementos (posiblemente ninguno) de una permutación de números \( (1, 2, \ldots, N) \). Cuando se elimina un elemento, se deja un espacio vacío (representado por \( 0 \)) en la secuencia y la longitud de la secuencia permanece igual. También se te da un entero \( K \).

Una permutación \( p \) se llama buena si:

- Es posible reemplazar los espacios vacíos en \( a \) por números de tal manera que obtengamos la permutación \( p \).
- El número de posiciones \( i \) (\( 1 < i \leq N \)) tales que \( p_i > p_{i-1} \) es igual a \( K \).

Tu tarea es encontrar el número de buenas permutaciones.

### Entrada

- La primera línea de la entrada contiene un solo entero \( T \), que denota el número de casos de prueba.
- Para cada caso de prueba:
  - La primera línea contiene dos enteros separados por espacios \( N \) y \( K \).
  - La segunda línea contiene \( N \) enteros separados por espacios \( a_1, a_2, \ldots, a_N \), donde cada elemento de esta secuencia es \( 0 \) (indicando un espacio vacío previamente ocupado por un elemento eliminado) o un entero entre 1 y \( N \) inclusive.

### Salida

Para cada caso de prueba, imprime una sola línea que contiene un entero: el número de buenas permutaciones.

### Ejemplo de Uso

#### Entrada

```
2
4 2
0 2 0 4
5 3
0 0 0 0 5
```

#### Salida

```
2
1
```

### Solución en Python

```python
def count_good_permutations(N, K, a):
    from itertools import permutations

    def is_good_permutation(p, K):
        count = 0
        for i in range(1, len(p)):
            if p[i] > p[i-1]:
                count += 1
        return count == K

    # Fill the empty spots (0s) with the remaining numbers
    available_numbers = set(range(1, N+1)) - set(a)
    empty_indices = [i for i in range(N) if a[i] == 0]

    good_permutations_count = 0

    for perm in permutations(available_numbers):
        b = a[:]
        for idx, val in zip(empty_indices, perm):
            b[idx] = val
        if is_good_permutation(b, K):
            good_permutations_count += 1

    return good_permutations_count

# Leer la entrada
T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    a = list(map(int, input().split()))
    result = count_good_permutations(N, K, a)
    print(result)
```

