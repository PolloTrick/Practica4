"""
Árbol Parcial Mínimo de Prim
============================
Encuentra el subconjunto de aristas que conecta todos los nodos
de un grafo ponderado con el menor costo total posible.

Cómo funciona:
  1. Empezar con un nodo arbitrario en el MST.
  2. En cada iteración, seleccionar la arista de menor peso
     que conecte un nodo dentro del MST con uno fuera.
  3. Agregar ese nodo y esa arista al MST.
  4. Repetir hasta que todos los nodos estén incluidos.

Complejidad:
  - Con lista de adyacencia + min-heap: O((V + E) log V)
  - Sólo funciona con grafos conexos y pesos no negativos.

Aplicaciones:
  Redes eléctricas, cableado de redes, planificación de rutas,
  diseño de circuitos, clustering en ML.

Uso rápido:
    g = Grafo()
    g.agregar_arista("A", "B", 4)
    g.prim("A")
"""

import heapq
from collections import defaultdict
from typing import Optional


class Grafo:
    """Grafo no dirigido ponderado con lista de adyacencia."""

    def __init__(self) -> None:
        # nodo -> [(peso, vecino), ...]
        self.adyacencia: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.nodos: set[str] = set()

    def agregar_arista(self, u: str, v: str, peso: int) -> None:
        """
        Agrega una arista bidireccional u-v con el peso dado.
        Lanza ValueError si el peso es negativo.
        """
        if peso < 0:
            raise ValueError(f"Peso negativo no permitido: ({u}, {v}, {peso})")
        self.adyacencia[u].append((peso, v))
        self.adyacencia[v].append((peso, u))
        self.nodos.update([u, v])

    # ------------------------------------------------------------------

    def prim(self, origen: str) -> list[tuple[str, str, int]]:
        """
        Ejecuta el algoritmo de Prim desde `origen` y retorna las
        aristas del Árbol Parcial Mínimo (MST).

        Estrategia:
          - Mantiene un conjunto `en_mst` con los nodos ya incluidos.
          - Usa un min-heap para seleccionar siempre la arista de
            menor peso que conecte el MST con el resto del grafo.
          - Al agregar un nodo, sus aristas hacia afuera se empujan
            al heap como nuevas candidatas.

        Parámetros:
            origen: Nodo de partida (cualquier nodo del grafo).

        Retorna:
            Lista de tuplas (u, v, peso) que forman el MST,
            en el orden en que fueron seleccionadas.

        Raises:
            KeyError: Si `origen` no existe en el grafo.
        """
        if origen not in self.nodos:
            raise KeyError(f"El nodo '{origen}' no existe en el grafo.")

        en_mst: set[str] = set()
        mst_aristas: list[tuple[str, str, int]] = []
        costo_total = 0

        # Heap: (peso, nodo_destino, nodo_origen)
        # nodo_origen es None para el nodo inicial
        heap: list[tuple[int, str, Optional[str]]] = [(0, origen, None)]

        # ── Cabecera consola ──────────────────────────────────────────
        print(f"\n{'='*58}")
        print(f"  Prim — MST desde '{origen}'")
        print(f"{'='*58}")
        print(f"  {'Paso':<5} {'Acción':<45} {'Costo'}")
        print(f"  {'-'*54}")

        paso = 1

        while heap:
            peso, nodo, desde = heapq.heappop(heap)

            # Ignorar si el nodo ya está en el MST
            if nodo in en_mst:
                print(f"  {'':5}   ↷ Ignorar '{desde}→{nodo}' (ya en MST)")
                continue

            # Agregar nodo al MST
            en_mst.add(nodo)

            if desde is not None:
                mst_aristas.append((desde, nodo, peso))
                costo_total += peso
                print(
                    f"  {paso:<5} ✔ Agregar arista {desde}→{nodo} "
                    f"(w={peso}){'':<10} {costo_total}"
                )
            else:
                print(f"  {paso:<5} ✔ Nodo inicial '{nodo}'")

            # Explorar vecinos del nodo recién agregado
            for w_vecino, vecino in self.adyacencia[nodo]:
                if vecino not in en_mst:
                    heapq.heappush(heap, (w_vecino, vecino, nodo))
                    print(
                        f"  {'':5}   → Candidata {nodo}→{vecino} (w={w_vecino})"
                    )

            paso += 1

        # ── Resumen ───────────────────────────────────────────────────
        print(f"  {'-'*54}")
        print(f"\n  Aristas del MST:")
        for u, v, w in mst_aristas:
            print(f"    {u} — {v}  (peso = {w})")
        print(f"\n  Costo total del MST: {costo_total}")

        if len(en_mst) < len(self.nodos):
            faltantes = self.nodos - en_mst
            print(f"  ⚠ Grafo no conexo. Nodos no alcanzados: {faltantes}")

        print(f"{'='*58}\n")
        return mst_aristas


# ---------------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------------

def ejemplo_ciudad() -> None:
    """
    Grafo tipo ciudad con 9 nodos (A–I) y 15 aristas.
    MST esperado conecta los 9 nodos con costo mínimo.
    """
    g = Grafo()
    for u, v, w in [
        ("A","B",4), ("A","C",2),
        ("B","C",1), ("B","D",5),
        ("C","D",8), ("C","E",10),
        ("D","E",2), ("D","F",6),
        ("E","F",3), ("E","G",7),
        ("F","G",1), ("F","H",4),
        ("G","H",2), ("G","I",5),
        ("H","I",1),
    ]:
        g.agregar_arista(u, v, w)
    g.prim("A")


def ejemplo_simple() -> None:
    """
    Grafo clásico de 6 nodos basado en el ejemplo de Dijkstra 1959.
    MST esperado tiene costo 25.
    """
    g = Grafo()
    for u, v, w in [
        ("A","B",7),  ("A","C",9),  ("A","F",14),
        ("B","C",10), ("B","D",15),
        ("C","D",11), ("C","F",2),
        ("D","E",6),  ("E","F",9),
    ]:
        g.agregar_arista(u, v, w)
    g.prim("A")


def ejemplo_complejo() -> None:
    """
    Grafo de 11 nodos (A–K) con 16 aristas.
    Útil para verificar que el heap maneja correctamente
    múltiples candidatas con pesos similares.
    """
    g = Grafo()
    for u, v, w in [
        ("A","B",2), ("A","C",6),
        ("B","D",5), ("B","E",8),
        ("C","D",3), ("C","F",7),
        ("D","G",4), ("E","G",2),
        ("E","H",9), ("F","G",1),
        ("F","I",5), ("G","H",3),
        ("G","J",6), ("H","K",4),
        ("I","J",2), ("J","K",3),
    ]:
        g.agregar_arista(u, v, w)
    g.prim("A")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "█"*58)
    print("  SIMULADOR ÁRBOL PARCIAL MÍNIMO — Algoritmo de Prim")
    print("█"*58)

    print("\n>>> Ejemplo 1: Grafo Ciudad (9 nodos)")
    ejemplo_ciudad()

    print(">>> Ejemplo 2: Grafo Clásico (6 nodos)")
    ejemplo_simple()

    print(">>> Ejemplo 3: Grafo Complejo (11 nodos)")
    ejemplo_complejo()