import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import time

# *** ════════════════════════════════════════════════════════════════════ ***
# *** NÚCLEO MATEMÁTICO - Transformaciones Lineales en R⁴ (Homogéneas) ***
# *** ════════════════════════════════════════════════════════════════════ ***

class TransformacionesLineales:
    """Biblioteca de matrices de transformación homogéneas 4x4."""
    # *** En este nivel de desarrollo, no solo aplicamos matemáticas; construimos un pipeline gráfico donde la interactividad se gestiona mediante un procesador de comandos algebraicos[cite: 146]. ***
    # *** Utilizamos coordenadas homogéneas (matrices de 4x4) para unificar todas las transformaciones, incluyendo la traslación, dentro del marco de aplicaciones lineales[cite: 136]. ***

    @staticmethod
    def traslacion(tx: float, ty: float, tz: float) -> np.ndarray:
        """T(tx,ty,tz): Traslación en coordenadas homogéneas."""
        # *** Se utiliza la cuarta dimensión ($w=1$) para permitir que la traslación se trate como una multiplicación matricial[cite: 137]. ***
        # *** Sin esto, la traslación sería una suma de vectores, rompiendo la elegancia del sistema matricial puro[cite: 138]. ***
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
        return T

    @staticmethod
    def escala(sx: float, sy: float, sz: float) -> np.ndarray:
        """S(sx,sy,sz): Escalamiento no uniforme."""
        # *** Escalamiento Dinámico: Mediante una matriz de escalamiento diagonal $S(s_x, s_y, s_z)$ para modificar el tamaño de forma precisa[cite: 169]. ***
        return np.diag([sx, sy, sz, 1.0])

    @staticmethod
    def rotacion_x(angulo_grados: float) -> np.ndarray:
        """Rx(θ): Rotación alrededor del eje X."""
        # *** Rotación de Alta Precisión: Utilizando matrices de rotación sobre el eje X calculadas con funciones trigonométricas de alta fidelidad de NumPy[cite: 170]. ***
        t = np.radians(angulo_grados)
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0,  0, 0],
                         [0, c, -s, 0],
                         [0, s,  c, 0],
                         [0, 0,  0, 1]], dtype=float)

    @staticmethod
    def rotacion_y(angulo_grados: float) -> np.ndarray:
        """Ry(θ): Rotación alrededor del eje Y."""
        # *** Rotación de Alta Precisión: Utilizando matrices de rotación sobre el eje Y calculadas con funciones trigonométricas de alta fidelidad de NumPy[cite: 170]. ***
        t = np.radians(angulo_grados)
        c, s = np.cos(t), np.sin(t)
        return np.array([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]], dtype=float)

    @staticmethod
    def rotacion_z(angulo_grados: float) -> np.ndarray:
        """Rz(θ): Rotación alrededor del eje Z."""
        # *** Rotación de Alta Precisión: Utilizando matrices de rotación sobre el eje Z calculadas con funciones trigonométricas de alta fidelidad de NumPy[cite: 170]. ***
        t = np.radians(angulo_grados)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]], dtype=float)

    @staticmethod
    def reflexion_plano(plano: str) -> np.ndarray:
        """Reflexión sobre planos XY, XZ o YZ."""
        # *** La reflexión es otra transformación lineal que invierte el signo de una coordenada en el espacio vectorial. ***
        M = np.eye(4)
        if plano == 'XY':   M[2, 2] = -1
        elif plano == 'XZ': M[1, 1] = -1
        elif plano == 'YZ': M[0, 0] = -1
        return M

    @staticmethod
    def cizallamiento(hxy=0, hxz=0, hyx=0, hyz=0, hzx=0, hzy=0) -> np.ndarray:
        """Shear/Cizallamiento: deforma el objeto de forma asimétrica."""
        # *** El cizallamiento desplaza cada capa del objeto proporcionalmente a su distancia de un eje, manteniendo el volumen constante en R³. ***
        return np.array([[1,   hyx, hzx, 0],
                         [hxy,  1,  hzy, 0],
                         [hxz, hyz,  1,  0],
                         [0,    0,   0,  1]], dtype=float)

    @staticmethod
    def componer(*matrices) -> np.ndarray:
        """Composición de transformaciones: M_n ∘ ... ∘ M_1"""
        # *** En el álgebra lineal, la composición de transformaciones demuestra la propiedad asociativa[cite: 120]. ***
        # *** Permite encadenar transformaciones simplemente multiplicando las matrices correspondientes antes de aplicarlas al objeto[cite: 120]. ***
        resultado = np.eye(4)
        for M in matrices:
            resultado = M @ resultado
        return resultado


# *** ════════════════════════════════════════════════════════════════════ ***
# *** MOTOR DE FÍSICA - Simulación con Euler + Colisión con piso ***
# *** ════════════════════════════════════════════════════════════════════ ***

class MotorFisica:
    """Simulador de física con integración de Euler."""
    # *** Física Vectorial: La gravedad se modela como un vector de traslación constante inyectado en el loop de animación[cite: 176]. ***
    # *** Esto demuestra que el movimiento es una serie de transformaciones lineales aplicadas en el tiempo[cite: 176]. ***

    GRAVEDAD = -9.8          # *** m/s² (ajustada para visualización) ***
    ELASTICIDAD = 0.65       # *** Coeficiente de restitución ***
    FRICCION = 0.92          # *** Factor de amortiguación horizontal ***
    PISO_Y = -6.0            # *** Altura del piso virtual ***

    def __init__(self):
        self.reset()

    def reset(self):
        self.activa = False
        self.velocidad = np.zeros(3)   # *** [vx, vy, vz] ***
        self.aceleracion = np.zeros(3) # *** [ax, ay, az] ***
        self.viento = np.zeros(3)
        self.spin = np.zeros(3)        # *** Velocidad angular durante física ***
        self.dt = 0.016                # *** Delta tiempo (60fps) ***
        self.en_piso = False

    def activar_gravedad(self):
        self.activa = True
        self.aceleracion = np.array([0.0, self.GRAVEDAD * 0.05, 0.0])

    def activar_impulso(self, fx, fy, fz):
        self.activa = True
        self.velocidad += np.array([fx, fy, fz])

    def activar_viento(self, vx, vy, vz):
        self.viento = np.array([vx, vy, vz]) * 0.02

    def paso(self, posicion_centro_y: float):
        """Retorna (delta_traslacion, delta_rotacion) para este frame."""
        if not self.activa:
            return np.zeros(3), np.zeros(3)

        # *** Integración de Euler: v += a*dt, p += v*dt ***
        # *** Física como Operador Lineal: La gravedad y el impulso se calculan como vectores de desplazamiento que se inyectan en la matriz de traslación en cada frame[cite: 159]. ***
        self.velocidad += (self.aceleracion + self.viento) * self.dt
        delta_pos = self.velocidad * self.dt

        # *** Detección de colisión con el piso ***
        # *** Colisiones Matemáticas: El sistema detecta el suelo evaluando el componente vertical del vector transformado, uniendo la geometría con la lógica[cite: 165]. ***
        if posicion_centro_y + delta_pos[1] <= self.PISO_Y:
            delta_pos[1] = self.PISO_Y - posicion_centro_y
            self.velocidad[1] *= -self.ELASTICIDAD
            self.velocidad[0] *= self.FRICCION
            self.velocidad[2] *= self.FRICCION
            self.en_piso = True
            # *** Spin al rebotar ***
            self.spin = np.array([
                self.velocidad[0] * 0.3,
                0,
                self.velocidad[2] * 0.3
            ])
        else:
            self.en_piso = False
            self.spin *= 0.95  # *** Amortiguación del spin ***

        return delta_pos, self.spin * self.dt


# *** ════════════════════════════════════════════════════════════════════ ***
# *** OBJETO 3D - Primitivas geométricas con soporte de mallas ***
# *** ════════════════════════════════════════════════════════════════════ ***

class Objeto3D:
    """Objeto geométrico con su propia matriz de transformación."""
    # *** Cada objeto se define como un subespacio en $\mathbb{R}^4$ usando coordenadas homogéneas $(x, y, z, 1)$[cite: 147]. ***

    PRIMITIVAS = {
        'cubo': {
            'vertices': np.array([
                [-1,-1,-1,1],[1,-1,-1,1],[1,1,-1,1],[-1,1,-1,1],
                [-1,-1, 1,1],[1,-1, 1,1],[1,1, 1,1],[-1,1, 1,1]
            ], dtype=float),
            'aristas': [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                        (0,4),(1,5),(2,6),(3,7)],
            'caras': [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
        },
        'piramide': {
            'vertices': np.array([
                [-1,-1,-1,1],[1,-1,-1,1],[1,-1,1,1],[-1,-1,1,1],[0,1.5,0,1]
            ], dtype=float),
            'aristas': [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)],
            'caras': [(0,1,2,3),(0,1,4),(1,2,4),(2,3,4),(3,0,4)]
        },
        'octaedro': {
            'vertices': np.array([
                [0,1.5,0,1],[1,0,0,1],[-1,0,0,1],[0,0,1,1],[0,0,-1,1],[0,-1.5,0,1]
            ], dtype=float),
            'aristas': [(0,1),(0,2),(0,3),(0,4),(5,1),(5,2),(5,3),(5,4),
                        (1,3),(3,2),(2,4),(4,1)],
            'caras': [(0,1,3),(0,3,2),(0,2,4),(0,4,1),(5,1,3),(5,3,2),(5,2,4),(5,4,1)]
        }
    }

    def __init__(self, tipo='cubo'):
        p = self.PRIMITIVAS[tipo]
        self.vertices_locales = p['vertices'].copy()
        self.aristas = p['aristas']
        self.caras = p['caras']
        self.M = np.eye(4)              # *** Matriz de transformación acumulada ***
        self.M_local = np.eye(4)        # *** Transformación base del objeto ***
        self.historial = []             # *** Stack de transformaciones ***
        self.color = '#00d1ff'
        self.color_cara = '#0a4a6e'

    def aplicar(self, T: np.ndarray, registrar=True):
        """Aplica T al espacio del objeto: M_nueva = T ∘ M"""
        # *** Matrices Acumulativas: El objeto guarda su estado en una única matriz $M$ que se actualiza mediante multiplicaciones sucesivas[cite: 174]. ***
        # *** Esto permite combinar rotación, escala y traslación sin perder precisión a lo largo del tiempo[cite: 174]. ***
        self.M = T @ self.M
        if registrar:
            self.historial.append(T.copy())

    def deshacer(self):
        """Revierte la última transformación aplicada."""
        if len(self.historial) > 1:
            self.historial.pop()
            self.M = np.eye(4)
            for T in self.historial:
                self.M = T @ self.M

    def reset(self):
        self.M = np.eye(4)
        self.historial = []

    def vertices_mundo(self) -> np.ndarray:
        """Aplica la transformación: v_mundo = M · v_local"""
        # *** Abstracción Lineal: Cada comando genera una matriz diferente, pero todas se aplican mediante la misma operación de transformación $T(v) = M \cdot v$[cite: 141]. ***
        # *** Se utiliza el operador @ de NumPy para realizar el producto punto de forma masiva sobre todos los vectores del objeto simultáneamente[cite: 140]. ***
        # *** Esto reduce la complejidad algorítmica procesando todos los vértices en una sola operación[cite: 118]. ***
        return (self.M @ self.vertices_locales.T).T

    def centro_mundo(self) -> np.ndarray:
        """Calcula el centroide del objeto en espacio mundo."""
        v = self.vertices_mundo()
        return np.mean(v[:, :3], axis=0)


# *** ════════════════════════════════════════════════════════════════════ ***
# *** INTERFAZ GRÁFICA PRINCIPAL ***
# *** ════════════════════════════════════════════════════════════════════ ***

COLORES = {
    'bg':        '#0d1117',
    'panel':     '#161b22',
    'acento':    '#58a6ff',
    'verde':     '#3fb950',
    'rojo':      '#f85149',
    'naranja':   '#d29922',
    'morado':    '#bc8cff',
    'texto':     '#e6edf3',
    'subtexto':  '#8b949e',
    'borde':     '#30363d',
}


class AppMotor(ctk.CTk):
    def __init__(self):
        super().__init__()

        # *** ── Configuración de la ventana ────────────────────────────── ***
        ctk.set_appearance_mode("dark")
        self.title("⬡ Motor Gráfico Algebraico — Transformaciones Lineales")
        self.geometry("1400x820")
        self.configure(fg_color=COLORES['bg'])

        # *** ── Estado del motor ───────────────────────────────────────── ***
        self.tl = TransformacionesLineales()
        self.objeto = Objeto3D('cubo')
        self.fisica = MotorFisica()
        self.mostrar_ejes = True
        self.mostrar_matriz = True
        self.mostrar_caras = True
        self.angulo_rotacion = 5.0
        self.escala_factor = 1.1
        self.tipo_objeto = 'cubo'
        self._ultimo_log = ""

        # *** ── Layout principal ───────────────────────────────────────── ***
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self._construir_panel_izquierdo()
        self._construir_viewport()
        self._construir_panel_derecho()

        # *** ── Inicio del loop de renderizado ─────────────────────────── ***
        self._update_loop()

    # *** ── PANELES UI ────────────────────────────────────────────────── ***

    def _construir_panel_izquierdo(self):
        panel = ctk.CTkScrollableFrame(
            self, width=290, fg_color=COLORES['panel'],
            scrollbar_fg_color=COLORES['panel'],
            scrollbar_button_color=COLORES['borde']
        )
        panel.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)

        # *** Título ***
        ctk.CTkLabel(panel, text="⬡ TRANSFORMACIONES",
                     font=("Courier New", 14, "bold"),
                     text_color=COLORES['acento']).pack(pady=(15,5))
        ctk.CTkLabel(panel, text="Álgebra Lineal · Espacio R³",
                     font=("Courier New", 10), text_color=COLORES['subtexto']).pack(pady=(0,15))

        # *** ── Objeto ──────────────────────────────────────────────── ***
        self._seccion(panel, "PRIMITIVA 3D")
        self.combo_objeto = ctk.CTkComboBox(
            panel, values=['cubo', 'piramide', 'octaedro'],
            command=self._cambiar_objeto,
            fg_color=COLORES['bg'], border_color=COLORES['borde'],
            button_color=COLORES['acento'], font=("Courier New", 12)
        )
        self.combo_objeto.set('cubo')
        self.combo_objeto.pack(padx=15, fill="x", pady=(0,10))

        # *** ── Traslación ──────────────────────────────────────────── ***
        self._seccion(panel, "TRASLACIÓN  T(tx, ty, tz)")
        self.entry_tx = self._slider_con_label(panel, "tx", -5, 5, 0)
        self.entry_ty = self._slider_con_label(panel, "ty", -5, 5, 0)
        self.entry_tz = self._slider_con_label(panel, "tz", -5, 5, 0)
        ctk.CTkButton(panel, text="▶  Aplicar Traslación",
                      fg_color=COLORES['acento'], text_color='black',
                      font=("Courier New", 12, "bold"),
                      command=self._trasladar).pack(padx=15, fill="x", pady=(5,15))

        # *** ── Escala ──────────────────────────────────────────────── ***
        self._seccion(panel, "ESCALAMIENTO  S(sx, sy, sz)")
        self.entry_sx = self._slider_con_label(panel, "sx", 0.1, 3, 1)
        self.entry_sy = self._slider_con_label(panel, "sy", 0.1, 3, 1)
        self.entry_sz = self._slider_con_label(panel, "sz", 0.1, 3, 1)
        ctk.CTkButton(panel, text="▶  Aplicar Escala",
                      fg_color=COLORES['verde'], text_color='black',
                      font=("Courier New", 12, "bold"),
                      command=self._escalar).pack(padx=15, fill="x", pady=(5,15))

        # *** ── Rotaciones ──────────────────────────────────────────── ***
        self._seccion(panel, "ROTACIÓN  R(θ)")
        row_rot = ctk.CTkFrame(panel, fg_color="transparent")
        row_rot.pack(padx=15, fill="x")
        self.entry_angulo = ctk.CTkEntry(row_rot, placeholder_text="θ°", width=70,
                                         fg_color=COLORES['bg'],
                                         border_color=COLORES['borde'],
                                         font=("Courier New", 12))
        self.entry_angulo.insert(0, "15")
        self.entry_angulo.pack(side="left", padx=(0,8))
        for eje, color in [("Rx", COLORES['rojo']), ("Ry", COLORES['verde']), ("Rz", COLORES['morado'])]:
            ctk.CTkButton(row_rot, text=eje, fg_color=color, text_color='black',
                          width=60, font=("Courier New", 11, "bold"),
                          command=lambda e=eje: self._rotar(e)).pack(side="left", padx=2)

        # *** ── Cizallamiento ───────────────────────────────────────── ***
        self._seccion(panel, "CIZALLAMIENTO  H(hxy)")
        self.entry_hxy = self._slider_con_label(panel, "hxy", -2, 2, 0)
        self.entry_hxz = self._slider_con_label(panel, "hxz", -2, 2, 0)
        ctk.CTkButton(panel, text="▶  Aplicar Cizalla",
                      fg_color=COLORES['naranja'], text_color='black',
                      font=("Courier New", 12, "bold"),
                      command=self._cizallar).pack(padx=15, fill="x", pady=(5,15))

        # *** ── Reflexión ───────────────────────────────────────────── ***
        self._seccion(panel, "REFLEXIÓN")
        row_ref = ctk.CTkFrame(panel, fg_color="transparent")
        row_ref.pack(padx=15, fill="x", pady=(0,15))
        for plano in ['XY', 'XZ', 'YZ']:
            ctk.CTkButton(row_ref, text=f"Ref {plano}",
                          fg_color=COLORES['morado'], text_color='black',
                          font=("Courier New", 11, "bold"), width=75,
                          command=lambda p=plano: self._reflejar(p)).pack(side="left", padx=3)

        # *** ── Controles ───────────────────────────────────────────── ***
        self._seccion(panel, "ACCIONES")
        ctk.CTkButton(panel, text="↩  Deshacer última T",
                      fg_color=COLORES['borde'], border_color=COLORES['acento'],
                      border_width=1, font=("Courier New", 12),
                      command=self._deshacer).pack(padx=15, fill="x", pady=3)
        ctk.CTkButton(panel, text="⟳  Resetear todo",
                      fg_color=COLORES['borde'], border_color=COLORES['rojo'],
                      border_width=1, font=("Courier New", 12),
                      command=self._reset).pack(padx=15, fill="x", pady=3)
        ctk.CTkButton(panel, text="⊞  Composición demo",
                      fg_color=COLORES['borde'], border_color=COLORES['naranja'],
                      border_width=1, font=("Courier New", 12),
                      command=self._demo_composicion).pack(padx=15, fill="x", pady=(3,15))

    def _construir_panel_derecho(self):
        panel = ctk.CTkFrame(self, width=310, fg_color=COLORES['panel'])
        panel.grid(row=0, column=2, sticky="nsew", padx=(5,10), pady=10)
        panel.grid_propagate(False)

        ctk.CTkLabel(panel, text="⬡ FÍSICA & INFO",
                     font=("Courier New", 14, "bold"),
                     text_color=COLORES['naranja']).pack(pady=(15,5))
        ctk.CTkLabel(panel, text="Simulación · Euler Integration",
                     font=("Courier New", 10), text_color=COLORES['subtexto']).pack(pady=(0,15))

        # *** ── Física ──────────────────────────────────────────────── ***
        self._seccion(panel, "SIMULACIÓN FÍSICA")

        self.btn_gravedad = ctk.CTkButton(
            panel, text="⬇  Activar Gravedad",
            fg_color=COLORES['verde'], text_color='black',
            font=("Courier New", 12, "bold"),
            command=self._toggle_gravedad
        )
        self.btn_gravedad.pack(padx=15, fill="x", pady=4)

        ctk.CTkLabel(panel, text="Impulso inicial (vx, vy, vz):",
                     font=("Courier New", 10), text_color=COLORES['subtexto']).pack(padx=15, anchor="w")
        self.entry_impulso = ctk.CTkEntry(panel, placeholder_text="0.5 3 0",
                                          fg_color=COLORES['bg'],
                                          border_color=COLORES['borde'],
                                          font=("Courier New", 12))
        self.entry_impulso.pack(padx=15, fill="x")
        ctk.CTkButton(panel, text="▶  Lanzar con impulso",
                      fg_color=COLORES['naranja'], text_color='black',
                      font=("Courier New", 11), command=self._impulso
                      ).pack(padx=15, fill="x", pady=4)

        ctk.CTkLabel(panel, text="Viento (vx, vy, vz):",
                     font=("Courier New", 10), text_color=COLORES['subtexto']).pack(padx=15, anchor="w")
        self.entry_viento = ctk.CTkEntry(panel, placeholder_text="1 0 0",
                                          fg_color=COLORES['bg'],
                                          border_color=COLORES['borde'],
                                          font=("Courier New", 12))
        self.entry_viento.pack(padx=15, fill="x")
        ctk.CTkButton(panel, text="≋  Aplicar viento",
                      fg_color=COLORES['borde'], border_color=COLORES['acento'],
                      border_width=1, font=("Courier New", 11),
                      command=self._viento).pack(padx=15, fill="x", pady=(4,15))

        # *** ── Opciones visuales ───────────────────────────────────── ***
        self._seccion(panel, "VISUALIZACIÓN")
        self.sw_ejes = ctk.CTkSwitch(panel, text="Mostrar ejes XYZ",
                                      font=("Courier New", 11),
                                      command=lambda: self._toggle('ejes'))
        self.sw_ejes.select()
        self.sw_ejes.pack(padx=15, anchor="w", pady=3)

        self.sw_caras = ctk.CTkSwitch(panel, text="Mostrar caras (semi-transparente)",
                                       font=("Courier New", 11),
                                       command=lambda: self._toggle('caras'))
        self.sw_caras.select()
        self.sw_caras.pack(padx=15, anchor="w", pady=3)

        self.sw_matriz = ctk.CTkSwitch(panel, text="Panel de matriz M",
                                        font=("Courier New", 11),
                                        command=lambda: self._toggle('matriz'))
        self.sw_matriz.select()
        self.sw_matriz.pack(padx=15, anchor="w", pady=(3,15))

        # *** ── Matriz en tiempo real ───────────────────────────────── ***
        self._seccion(panel, "MATRIZ DE TRANSFORMACIÓN M")
        self.lbl_matriz = ctk.CTkLabel(panel, text="",
                                        font=("Courier New", 11),
                                        text_color=COLORES['acento'],
                                        justify="left")
        self.lbl_matriz.pack(padx=15, anchor="w", pady=5)

        # *** ── Log de operaciones ──────────────────────────────────── ***
        self._seccion(panel, "HISTORIAL DE OPERACIONES")
        self.txt_log = ctk.CTkTextbox(panel, height=180,
                                       font=("Courier New", 10),
                                       fg_color=COLORES['bg'],
                                       border_color=COLORES['borde'],
                                       border_width=1,
                                       text_color=COLORES['texto'])
        self.txt_log.pack(padx=15, fill="x", pady=(0,15))

        # *** ── Info física en tiempo real ──────────────────────────── ***
        self._seccion(panel, "ESTADO FÍSICO")
        self.lbl_fisica = ctk.CTkLabel(panel, text="",
                                        font=("Courier New", 10),
                                        text_color=COLORES['verde'],
                                        justify="left")
        self.lbl_fisica.pack(padx=15, anchor="w", pady=5)

    def _construir_viewport(self):
        frame = ctk.CTkFrame(self, fg_color=COLORES['bg'])
        frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # *** Barra de título del viewport ***
        barra = ctk.CTkFrame(frame, height=35, fg_color=COLORES['panel'])
        barra.grid(row=0, column=0, sticky="ew", padx=0, pady=(0,5))
        ctk.CTkLabel(barra, text="  VIEWPORT 3D  ·  Proyección perspectiva  ·  60fps",
                     font=("Courier New", 11), text_color=COLORES['subtexto']).pack(side="left")

        self.lbl_fps = ctk.CTkLabel(barra, text="fps: --",
                                     font=("Courier New", 11), text_color=COLORES['verde'])
        self.lbl_fps.pack(side="right", padx=10)

        # *** Figura matplotlib ***
        # *** Integramos NumPy y Matplotlib para el procesamiento masivo de datos vectoriales y renderizado[cite: 172]. ***
        plt.style.use('dark_background')
        self.fig = plt.figure(facecolor=COLORES['bg'], figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self._t_prev = time.time()

    # *** ── HELPERS UI ────────────────────────────────────────────────── ***

    def _seccion(self, parent, texto):
        f = ctk.CTkFrame(parent, fg_color=COLORES['borde'], height=1)
        f.pack(padx=15, fill="x", pady=(10,0))
        ctk.CTkLabel(parent, text=texto, font=("Courier New", 10, "bold"),
                     text_color=COLORES['subtexto']).pack(padx=15, anchor="w", pady=(3,5))

    def _slider_con_label(self, parent, nombre, minv, maxv, default):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(padx=15, fill="x", pady=2)
        lbl = ctk.CTkLabel(frame, text=f"{nombre}:", width=30,
                            font=("Courier New", 11), text_color=COLORES['texto'])
        lbl.pack(side="left")
        val_var = ctk.StringVar(value=str(default))
        lbl_val = ctk.CTkLabel(frame, textvariable=val_var, width=45,
                                font=("Courier New", 11), text_color=COLORES['acento'])
        lbl_val.pack(side="right")
        slider = ctk.CTkSlider(frame, from_=minv, to=maxv,
                                fg_color=COLORES['borde'],
                                progress_color=COLORES['acento'],
                                button_color=COLORES['acento'],
                                command=lambda v, lv=lbl_val, vv=val_var: vv.set(f"{v:.2f}"))
        slider.set(default)
        slider.pack(side="left", fill="x", expand=True, padx=5)
        frame._slider = slider
        return frame

    def _get_slider(self, frame):
        return frame._slider.get()

    # *** ── ACCIONES DE TRANSFORMACIÓN ────────────────────────────────── ***

    def _trasladar(self):
        tx = self._get_slider(self.entry_tx)
        ty = self._get_slider(self.entry_ty)
        tz = self._get_slider(self.entry_tz)
        T = self.tl.traslacion(tx, ty, tz)
        self.objeto.aplicar(T)
        self._log(f"T({tx:.2f}, {ty:.2f}, {tz:.2f})")

    def _escalar(self):
        sx = self._get_slider(self.entry_sx)
        sy = self._get_slider(self.entry_sy)
        sz = self._get_slider(self.entry_sz)
        S = self.tl.escala(sx, sy, sz)
        self.objeto.aplicar(S)
        self._log(f"S({sx:.2f}, {sy:.2f}, {sz:.2f})")

    def _rotar(self, eje):
        try:
            angulo = float(self.entry_angulo.get())
        except ValueError:
            angulo = 15.0
        if eje == 'Rx':
            R = self.tl.rotacion_x(angulo)
        elif eje == 'Ry':
            R = self.tl.rotacion_y(angulo)
        else:
            R = self.tl.rotacion_z(angulo)
        self.objeto.aplicar(R)
        self._log(f"{eje}({angulo:.1f}°)")

    def _reflejar(self, plano):
        F = self.tl.reflexion_plano(plano)
        self.objeto.aplicar(F)
        self._log(f"Ref({plano})")

    def _cizallar(self):
        hxy = self._get_slider(self.entry_hxy)
        hxz = self._get_slider(self.entry_hxz)
        H = self.tl.cizallamiento(hxy=hxy, hxz=hxz)
        self.objeto.aplicar(H)
        self._log(f"H(hxy={hxy:.2f}, hxz={hxz:.2f})")

    def _deshacer(self):
        self.objeto.deshacer()
        self._log("↩ Deshacer")

    def _reset(self):
        self.objeto.reset()
        self.fisica.reset()
        self._log("⟳ Reset completo")

    def _demo_composicion(self):
        """Demuestra T∘R∘S: primero escala, luego rota, luego traslada."""
        S = self.tl.escala(0.7, 1.4, 0.7)
        R = self.tl.rotacion_y(45)
        T = self.tl.traslacion(2, 1, 0)
        M = self.tl.componer(T, R, S)
        self.objeto.aplicar(M)
        self._log("Demo: T∘Ry(45)∘S(0.7,1.4,0.7)")

    def _cambiar_objeto(self, tipo):
        self.tipo_objeto = tipo
        self.objeto = Objeto3D(tipo)
        self.fisica.reset()
        self._log(f"Objeto: {tipo}")

    # *** ── FÍSICA ────────────────────────────────────────────────────── ***

    def _toggle_gravedad(self):
        if self.fisica.activa:
            self.fisica.reset()
            self.btn_gravedad.configure(text="⬇  Activar Gravedad",
                                        fg_color=COLORES['verde'])
            self._log("Física: desactivada")
        else:
            self.fisica.activar_gravedad()
            self.btn_gravedad.configure(text="⬛  Detener Física",
                                        fg_color=COLORES['rojo'])
            self._log("Física: gravedad activada")

    def _impulso(self):
        try:
            vals = [float(x) for x in self.entry_impulso.get().split()]
            if len(vals) == 3:
                self.fisica.activar_impulso(*vals)
                self.fisica.activar_gravedad()
                self.btn_gravedad.configure(text="⬛  Detener Física",
                                            fg_color=COLORES['rojo'])
                self._log(f"Impulso({vals[0]},{vals[1]},{vals[2]})")
        except Exception:
            pass

    def _viento(self):
        try:
            vals = [float(x) for x in self.entry_viento.get().split()]
            if len(vals) == 3:
                self.fisica.activar_viento(*vals)
                self._log(f"Viento({vals[0]},{vals[1]},{vals[2]})")
        except Exception:
            pass

    # *** ── OPCIONES VISUALES ─────────────────────────────────────────── ***

    def _toggle(self, opcion):
        if opcion == 'ejes':    self.mostrar_ejes = self.sw_ejes.get()
        elif opcion == 'caras': self.mostrar_caras = self.sw_caras.get()
        elif opcion == 'matriz': self.mostrar_matriz = self.sw_matriz.get()

    def _log(self, texto):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"▸ {texto}\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")
        self._ultimo_log = texto

    # *** ── LOOP DE RENDERIZADO ───────────────────────────────────────── ***

    def _update_loop(self):
        # *** Paso físico ***
        # *** Procesamos las físicas como transformaciones de traslación acumulativas[cite: 157]. ***
        if self.fisica.activa:
            centro_y = self.objeto.centro_mundo()[1]
            delta_pos, delta_rot = self.fisica.paso(centro_y)
            if np.any(np.abs(delta_pos) > 1e-6):
                T = self.tl.traslacion(*delta_pos)
                self.objeto.aplicar(T, registrar=False)
            if np.any(np.abs(delta_rot) > 1e-6):
                Rx = self.tl.rotacion_x(np.degrees(delta_rot[0]))
                Rz = self.tl.rotacion_z(np.degrees(delta_rot[2]))
                self.objeto.aplicar(Rx @ Rz, registrar=False)

        # *** Renderizado ***
        self._render()

        # *** Actualizar info lateral ***
        self._actualizar_info()

        self.after(16, self._update_loop)

    def _render(self):
        t_now = time.time()
        fps = 1.0 / max(t_now - self._t_prev, 1e-9)
        self._t_prev = t_now
        self.lbl_fps.configure(text=f"fps: {fps:.0f}")

        self.ax.clear()
        self.ax.set_facecolor(COLORES['bg'])

        # *** Proyectamos los vectores del espacio 3D al plano 2D de la pantalla[cite: 160]. ***
        v = self.objeto.vertices_mundo()
        xs, ys, zs = v[:, 0], v[:, 1], v[:, 2]

        # *** ── Piso virtual ── ***
        if self.fisica.activa:
            px = np.array([-8, 8, 8, -8, -8])
            pz = np.array([-8, -8, 8, 8, -8])
            py = np.full(5, self.fisica.PISO_Y)
            self.ax.plot(px, py, pz, color='#333344', alpha=0.5)
            self.ax.plot_surface(
                np.array([[-8,8],[-8,8]]),
                np.array([[self.fisica.PISO_Y]*2]*2),
                np.array([[-8,-8],[8,8]]),
                alpha=0.12, color='#4455aa'
            )

        # *** ── Caras semitransparentes ── ***
        if self.mostrar_caras and len(self.objeto.caras) > 0:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poligonos = []
            for cara in self.objeto.caras:
                poligonos.append([v[i, :3] for i in cara])
            poly = Poly3DCollection(poligonos, alpha=0.15,
                                    facecolor='#1a4a8a', edgecolor='none')
            self.ax.add_collection3d(poly)

        # *** ── Aristas ── ***
        for a, b in self.objeto.aristas:
            self.ax.plot(
                [xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                color=COLORES['acento'], linewidth=1.8, alpha=0.95
            )

        # *** ── Vértices ── ***
        self.ax.scatter(xs, ys, zs, color=COLORES['acento'],
                        s=20, alpha=0.9, zorder=5)

        # *** ── Ejes del mundo ── ***
        if self.mostrar_ejes:
            lim = 7
            self.ax.quiver(0,0,0, lim,0,0, color='#ff4444', linewidth=1.5, arrow_length_ratio=0.1)
            self.ax.quiver(0,0,0, 0,lim,0, color='#44ff44', linewidth=1.5, arrow_length_ratio=0.1)
            self.ax.quiver(0,0,0, 0,0,lim, color='#4444ff', linewidth=1.5, arrow_length_ratio=0.1)
            self.ax.text(lim+0.3, 0, 0, 'X', color='#ff4444', fontsize=9, fontweight='bold')
            self.ax.text(0, lim+0.3, 0, 'Y', color='#44ff44', fontsize=9, fontweight='bold')
            self.ax.text(0, 0, lim+0.3, 'Z', color='#4444ff', fontsize=9, fontweight='bold')

        # *** ── Límites y estilo ── ***
        RANGO = 10
        self.ax.set_xlim([-RANGO, RANGO])
        self.ax.set_ylim([-RANGO, RANGO])
        self.ax.set_zlim([-RANGO, RANGO])
        self.ax.set_xlabel('X', color=COLORES['subtexto'], fontsize=8)
        self.ax.set_ylabel('Y', color=COLORES['subtexto'], fontsize=8)
        self.ax.set_zlabel('Z', color=COLORES['subtexto'], fontsize=8)
        self.ax.tick_params(colors=COLORES['subtexto'], labelsize=7)
        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(COLORES['borde'])
        self.ax.grid(True, color=COLORES['borde'], alpha=0.3, linestyle='--')

        # *** ── Título del viewport con última transformación ── ***
        self.ax.set_title(
            f"  {self._ultimo_log}" if self._ultimo_log else "  Motor Gráfico Algebraico",
            color=COLORES['subtexto'], fontsize=9, pad=8, loc='left',
            fontfamily='monospace'
        )

        self.canvas.draw()

    def _actualizar_info(self):
        # *** Matriz de transformación ***
        M = self.objeto.M
        lineas = []
        for i in range(4):
            fila = "  ".join(f"{M[i,j]:+.2f}" for j in range(4))
            lineas.append(f"│ {fila} │")
        self.lbl_matriz.configure(text="\n".join(lineas))

        # *** Estado físico ***
        if self.fisica.activa:
            v = self.fisica.velocidad
            speed = np.linalg.norm(v)
            estado = (
                f"v = ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})\n"
                f"|v| = {speed:.2f} u/s\n"
                f"En piso: {'Sí ⬛' if self.fisica.en_piso else 'No'}\n"
                f"Gravedad: {self.fisica.GRAVEDAD} m/s²"
            )
        else:
            c = self.objeto.centro_mundo()
            estado = (
                f"Centro: ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})\n"
                f"Trans. acum.: {len(self.objeto.historial)}\n"
                f"Física: inactiva"
            )
        self.lbl_fisica.configure(text=estado)


# *** ════════════════════════════════════════════════════════════════════ ***
# *** ENTRY POINT ***
# *** ════════════════════════════════════════════════════════════════════ ***

if __name__ == "__main__":
    app = AppMotor()
    app.mainloop()