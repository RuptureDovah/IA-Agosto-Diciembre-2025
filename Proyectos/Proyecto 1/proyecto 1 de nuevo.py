import pygame
from queue import PriorityQueue
import math

# -----------------------
# Configuraciones editables (cambia aquí)
# -----------------------
GRID_SIZE = 11         # <-- Cambia este valor para elegir el tamaño de la matriz (filas x filas)
ANCHO_INICIAL = 800
ALTO_INICIAL = 600
ALTURA_EXTRA = 70       # altura de la franja inferior para botones
# -----------------------

# Inicializar pygame
pygame.init()
VENTANA = pygame.display.set_mode((ANCHO_INICIAL, ALTO_INICIAL), pygame.RESIZABLE)
pygame.display.set_caption("Visualización de A*")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (180, 180, 180)
VERDE = (0, 200, 0)
ROJO = (200, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 120, 255)
NEGRO_CLARO = (40, 40, 40)

# ========================
# UI: botones (posiciones relativas)
# ========================
def dibujar_boton(ventana, texto, rect, color, fuente):
    pygame.draw.rect(ventana, color, rect, border_radius=6)
    superficie_texto = fuente.render(texto, True, BLANCO)
    txt_rect = superficie_texto.get_rect(center=rect.center)
    ventana.blit(superficie_texto, txt_rect)

def dentro_boton_pos(pos, rect):
    return rect.collidepoint(pos)

# ========================
# Clase Nodo
# ========================
class Nodo:
    def __init__(self, fila, col, total_filas, index):
        self.fila = int(fila)
        self.col = int(col)
        self.index = int(index)   # número secuencial 1..N
        self.x = 0
        self.y = 0
        self.color = BLANCO
        self.vecinos = []
        self.ancho = 0
        self.total_filas = total_filas

    def __eq__(self, other):
        return isinstance(other, Nodo) and (self.fila, self.col) == (other.fila, other.col)

    def __hash__(self):
        return hash((self.fila, self.col))

    def set_pos_pixel(self, ancho_nodo, offset_x=0, offset_y=0):
        """Actualizar x,y y ancho (se llama en cada dibujado para adaptarse al resize)."""
        self.ancho = ancho_nodo
        self.x = self.col * ancho_nodo + offset_x
        self.y = self.fila * ancho_nodo + offset_y

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_visitado(self):
        self.color = VERDE

    def hacer_abierto(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = AZUL

    def actualizar_vecinos(self, grid):
        """Rellena self.vecinos basándose en la cuadrícula, evitando paredes.
        Incluye diagonales y evita corner-cutting (requiere que ambos ortogonales estén libres)."""
        self.vecinos = []
        filas = self.total_filas
        r, c = self.fila, self.col

        # movimientos ortogonales (dr,dc,cost)
        dirs_ort = [ (1,0), (-1,0), (0,1), (0,-1) ]
        for dr, dc in dirs_ort:
            nr, nc = r + dr, c + dc
            if 0 <= nr < filas and 0 <= nc < filas and not grid[nr][nc].es_pared():
                self.vecinos.append((grid[nr][nc], 1.0))

        # diagonales (dr,dc) con coste sqrt(2); evitar corner-cutting:
        dirs_diag = [ (1,1), (1,-1), (-1,1), (-1,-1) ]
        for dr, dc in dirs_diag:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < filas and 0 <= nc < filas):
                continue
            # comprobar que la diagonal no atraviese una esquina ocupada:
            # requerimos que las celdas ortogonales adyacentes estén libres
            ort1 = grid[r + dr][c]   # vertical adyacente
            ort2 = grid[r][c + dc]   # horizontal adyacente
            diag = grid[nr][nc]
            if not diag.es_pared() and not ort1.es_pared() and not ort2.es_pared():
                self.vecinos.append((diag, math.sqrt(2)))

# ========================
# Heurística y reconstrucción
# ========================
def h(p1, p2):
    # Usamos distancia Euclidiana para heurística (admisible con costes ort/diag)
    x1, y1 = p1
    x2, y2 = p2
    return abs(y2-y1)+abs(x2-x1)  

def reconstruir_camino(came_from, actual, dibujar):
    # Devuelve lista de nodos (desde inicio -> ... -> nodo previo al 'actual' según came_from)
    path_nodes = []
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        path_nodes.append(actual)
        dibujar()
    path_nodes.reverse()
    return path_nodes

# ========================
# A* (devuelve tuple: (lista_indices, costo_total) o None)
# ========================
def algoritmo_a_star(dibujar, grid, inicio, fin):
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    came_from = {}
    # inicializar g_score y f_score
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0.0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}

    while not open_set.empty():
        # permitir cerrar la ventana durante la búsqueda
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        actual = open_set.get()[2]
        # si venía en el hash lo removemos (si no estaba, ya fue procesado)
        if actual in open_set_hash:
            open_set_hash.remove(actual)

        if actual == fin:
            # reconstruir camino y devolver índices + costo total
            camino_parcial = reconstruir_camino(came_from, fin, dibujar)  # nodos desde inicio(excl) -> nodo antes de fin
            camino_nodes = [inicio] + camino_parcial + [fin]
            indices = [n.index for n in camino_nodes]
            costo_total = g_score[fin]
            # asegurar colores
            fin.hacer_fin()
            inicio.hacer_inicio()
            return indices, costo_total

        # cada vecino ahora es (Nodo, coste_movimiento)
        for vecino, move_cost in actual.vecinos:
            temp_g_score = g_score[actual] + move_cost

            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((f_score[vecino], contador, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar()

        if actual != inicio:
            actual.hacer_visitado()

    return None

# ========================
# Grid y dibujo
# ========================
def crear_grid(filas):
    grid = []
    index = 1
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            grid[i].append(Nodo(i, j, filas, index))
            index += 1
    return grid

def dibujar(ventana, grid, filas, ancho_total, alto_total, fuente_coord, fuente_btn):
    ventana.fill(BLANCO)

    # área del grid cuadrada (lado)
    ancho_area = min(ancho_total, max(0, alto_total - ALTURA_EXTRA))
    offset_x = (ancho_total - ancho_area) // 2
    offset_y = 0

    # dibujar fondo del area
    pygame.draw.rect(ventana, BLANCO, (offset_x, offset_y, ancho_area, ancho_area))

    # tamaño nodo (int)
    ancho_nodo = max(1, ancho_area // filas)

    # dibujar nodos
    for fila in grid:
        for nodo in fila:
            nodo.set_pos_pixel(ancho_nodo, offset_x, offset_y)
            rect = pygame.Rect(nodo.x, nodo.y, nodo.ancho, nodo.ancho)
            pygame.draw.rect(ventana, nodo.color, rect)
            pygame.draw.rect(ventana, NEGRO_CLARO, rect, 1)

            # Texto: mostramos index secuencial.
            texto = f"{nodo.index}"
            surf = fuente_coord.render(texto, True, NEGRO_CLARO)
            trect = surf.get_rect(center=(rect.x + rect.w//2, rect.y + rect.h//2))
            ventana.blit(surf, trect)

    # líneas de la cuadrícula
    for i in range(filas + 1):
        y = offset_y + i * ancho_nodo
        pygame.draw.line(ventana, GRIS, (offset_x, y), (offset_x + ancho_area, y))
    for j in range(filas + 1):
        x = offset_x + j * ancho_nodo
        pygame.draw.line(ventana, GRIS, (x, offset_y), (x, offset_y + ancho_area))

    # Barra inferior y botones: centramos los botones dentro de la franja ALTURA_EXTRA
    franja_top = offset_y + ancho_area
    boton_h = 40
    boton_w = max(100, ancho_total // 6)
    espacio_vertical = max( (ALTURA_EXTRA - boton_h) // 2, 5)
    y_botones = franja_top + espacio_vertical

    margin = 30
    available_w = ancho_total - 2 * margin
    gap = max(20, (available_w - 3 * boton_w) // 2)
    x_iniciar = margin
    x_reiniciar = margin + boton_w + gap
    x_salir = margin + 2 * (boton_w + gap)

    if x_salir + boton_w + margin > ancho_total:
        boton_w = max(60, (ancho_total - 2 * margin - 2 * gap) // 3)
        x_iniciar = margin
        x_reiniciar = margin + boton_w + gap
        x_salir = margin + 2 * (boton_w + gap)

    rect_iniciar = pygame.Rect(x_iniciar, y_botones, boton_w, boton_h)
    rect_reiniciar = pygame.Rect(x_reiniciar, y_botones, boton_w, boton_h)
    rect_salir = pygame.Rect(x_salir, y_botones, boton_w, boton_h)

    dibujar_boton(ventana, "INICIAR", rect_iniciar, VERDE, fuente_btn)
    dibujar_boton(ventana, "REINICIAR", rect_reiniciar, NARANJA, fuente_btn)
    dibujar_boton(ventana, "SALIR", rect_salir, ROJO, fuente_btn)

    pygame.display.update()

    return {'iniciar': rect_iniciar, 'reiniciar': rect_reiniciar, 'salir': rect_salir}

# ========================
# Convierte click -> fila/col usando offsets
# ========================
def obtener_click_pos(pos, filas, ancho_total, alto_total):
    x_mouse, y_mouse = pos
    ancho_area = min(ancho_total, max(0, alto_total - ALTURA_EXTRA))
    offset_x = (ancho_total - ancho_area) // 2
    offset_y = 0
    ancho_nodo = max(1, ancho_area // filas)

    if x_mouse < offset_x or x_mouse >= offset_x + ancho_area or y_mouse < offset_y or y_mouse >= offset_y + ancho_area:
        return None, None

    col = (x_mouse - offset_x) // ancho_nodo
    fila = (y_mouse - offset_y) // ancho_nodo
    fila = max(0, min(fila, filas - 1))
    col = max(0, min(col, filas - 1))
    return fila, col

# ========================
# Main
# ========================
def main(ventana):
    FILAS = GRID_SIZE
    grid = crear_grid(FILAS)

    inicio = None
    fin = None
    corriendo = True
    clock = pygame.time.Clock()

    fuente_btn = pygame.font.SysFont("arial", 18)
    fuente_coord = pygame.font.SysFont("arial", 14)

    button_rects = {}

    while corriendo:
        clock.tick(60)
        ancho_total, alto_total = pygame.display.get_surface().get_size()

        # ajustar fuentes según tamaño de celda
        ancho_area = min(ancho_total, max(0, alto_total - ALTURA_EXTRA))
        ancho_nodo = max(1, ancho_area // FILAS)
        fuente_coord = pygame.font.SysFont("arial", max(8, ancho_nodo // 3))
        fuente_btn = pygame.font.SysFont("arial", max(12, ancho_nodo // 2))

        # dibujar
        button_rects = dibujar(
            ventana, grid, FILAS,
            ancho_total, alto_total,
            fuente_coord, fuente_btn
        )

        # EVENTOS (clicks, teclado, resize)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            elif event.type == pygame.VIDEORESIZE:
                ventana = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                if dentro_boton_pos(pos, button_rects['iniciar']):
                    if inicio and fin:
                        for fila in grid:
                            for nodo in fila:
                                nodo.actualizar_vecinos(grid)

                        result = algoritmo_a_star(
                            lambda: dibujar(
                                ventana, grid, FILAS,
                                ancho_total, alto_total,
                                fuente_coord, fuente_btn
                            ),
                            grid, inicio, fin
                        )

                        if result:
                            indices, costo = result
                            print("Camino:", indices)
                            print("Costo:", f"{costo:.3f}")

                elif dentro_boton_pos(pos, button_rects['reiniciar']):
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS)

                elif dentro_boton_pos(pos, button_rects['salir']):
                    corriendo = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS)

                elif event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    algoritmo_a_star(
                        lambda: dibujar(
                            ventana, grid, FILAS,
                            ancho_total, alto_total,
                            fuente_coord, fuente_btn
                        ),
                        grid, inicio, fin
                    )

        #mouse continuo para dibujar paredes
        mouse = pygame.mouse.get_pressed()
        pos = pygame.mouse.get_pos()
        fila, col = obtener_click_pos(pos, FILAS, ancho_total, alto_total)

        if fila is not None:
            nodo = grid[fila][col]

            if mouse[0]:  # izquierdo
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != inicio and nodo != fin:
                    nodo.hacer_pared()

            elif mouse[2]:  # derecho
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

    pygame.quit()


# Ejecutar main
main(VENTANA)
