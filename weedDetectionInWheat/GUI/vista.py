from customtkinter import *
from tkinter import filedialog
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import cv2
import random
import numpy as np

class Ventana():

    def __init__(self):
        
        self.app = CTk()
        self.ventanaAncho = self.obtenerAnchoPantala(self.app, 60)
        self.ventanaLargo = self.obtenerLargoPantalla(self.app, 60)
        self.app.title("Clasificador Imágenes")
        self.app.after(201, lambda: self.app.iconbitmap("weedDetectionInWheat/GUI/Imagenes/Icono.ico"))

        self.app.geometry(f"{self.ventanaAncho}x{self.ventanaLargo}")

        self.colorFondo = '#373739'
        self.primerGris = "#19191a"
        self.segundoGris = "#4A4A4D"
        self.textoGris = "#c6c6c6"
        self.verde = "#4FC723"
        self.verdeHover = "#006600"

        self.modelo = keras.models.load_model("weedDetectionInWheat/CNN/alexnetNormalized2.keras")
        self.ruta = None

        self.generarCotenedores(self.app, self.ventanaAncho, self.ventanaLargo)

        self.app.mainloop()
    
    def obtenerAnchoPantala(self, Ventana, Porcentaje, Ancho = None):

        if(Ancho is None):

            Ancho = Ventana.winfo_screenwidth()

        return Ancho * Porcentaje // 100
    
    def obtenerLargoPantalla(self, Ventana, Porcentaje, Largo = None):

        if (Largo is None):

            Largo = Ventana.winfo_screenheight()
        
        return Largo * Porcentaje // 100
    
    def obtenerAncho(self, Porcentaje, Ancho):
        
        return Ancho * Porcentaje // 100
    
    def obtenerLargo(self, Porcentaje, Largo):

        return Largo * Porcentaje // 100
    
    def generarCotenedores(self, Ventana, Ancho, Largo):

        Ventana.rowconfigure(0, weight = 1)
        Ventana.rowconfigure(1, weight = 1)
        Ventana.columnconfigure(0, weight = 1)

        imagenesAncho = self.obtenerAncho(100, Ancho)
        imagenesLargo = self.obtenerLargo(80, Largo)

        resultadosAncho = self.obtenerAncho(100, Ancho)
        resultadosLargo = self.obtenerLargo(20, Largo)

        imagenes = CTkFrame(master = Ventana,
                            width = imagenesAncho,
                            height = imagenesLargo,
                            fg_color = self.primerGris,
                            corner_radius = 0,
                            border_width = 0)
        
        resultados = CTkFrame(master = Ventana,
                              width = resultadosAncho,
                              height = resultadosLargo,
                              fg_color = self.segundoGris,
                              corner_radius = 0,
                              border_width = 0)
        
        imagenes.grid(row = 0, column = 0, sticky = "nsew")
        resultados.grid(row = 1, column = 0, sticky = "nsew")

        self.generarImagenes(imagenes, imagenesAncho, imagenesLargo)
        self.generarLabels(resultados, resultadosAncho, resultadosLargo)

    def generarImagenes(self, frame, frameAncho, frameLargo,):

        self.imagenesFrame = frame
        self.imagenesAnchoFrame = frameAncho
        self.imagenesLargoFrame = frameLargo

        imagen = Image.open("weedDetectionInWheat/GUI/Imagenes/Planta.png")
        imagenAncho, imagenLargo = imagen.size

        imagenAncho = self.obtenerAncho(imagenAncho, 30)
        imagenLargo = self.obtenerAncho(imagenLargo, 30)
        imagenCTK = CTkImage(imagen, size=(imagenAncho, imagenLargo))

        contenedorImagen = CTkLabel(master=frame,
                                    text="",
                                    image=imagenCTK)
        contenedorImagen.place(relx=0.5, rely=0.4, anchor="center")

        anchoBoton = self.obtenerAncho(frameAncho, 20)
        largoBoton = self.obtenerLargo(frameLargo, 10)

        botonArchivos = CTkButton(master=frame,
                                width=anchoBoton,
                                height=largoBoton,
                                corner_radius = 0,
                                text="Seleccionar Carpeta",
                                fg_color=self.verde,
                                hover_color=self.verdeHover,
                                font=("Helvetica", 16),
                                command=self.iniciarPrediccion)
        botonArchivos.place(relx=0.5, rely=0.7, anchor="center")

    def generarLabels(self, frame, anchoFrame, largoFrame):

        self.textoFrame = frame

        esperado = "Esperado: "
        prediccion = "Predicción: "

        labelEsperado = CTkLabel(master=frame, text=esperado, font=("Helvetica", 16), text_color=self.textoGris, anchor="w")
        labelEsperado.place(relx=0.2, rely=0.6, anchor="w")
        self.valorEsperado = CTkLabel(master=frame, text="0", font=("Helvetica", 16), text_color=self.textoGris, anchor="w")
        self.valorEsperado.place(relx=0.6, rely=0.6, anchor="w")

        labelPrediccion = CTkLabel(master=frame, text=prediccion, font=("Helvetica", 16), text_color=self.textoGris, anchor="w")
        labelPrediccion.place(relx=0.2, rely=0.3, anchor="w")
        self.valorPrediccion = CTkLabel(master=frame, text="0", font=("Helvetica", 16), text_color=self.textoGris, anchor="w")
        self.valorPrediccion.place(relx=0.6, rely=0.3, anchor="w")
        self.valorPorcentaje = CTkLabel(master=frame, text="0.0 %", font=("Helvetica", 16), text_color=self.textoGris, anchor="w")
        self.valorPorcentaje.place(relx=0.8, rely=0.3, anchor="w")
        
    def iniciarPrediccion(self):

        self.ruta = filedialog.askdirectory()

        if self.ruta:
        
            for widget in self.imagenesFrame.winfo_children():
                widget.destroy()

        formatosValidos = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        self.imagenes = []

        for root, dirs, files in os.walk(self.ruta):

            for f in files:

                if f.lower().endswith(formatosValidos):

                    rutaCompleta = os.path.join(root, f)
                    self.imagenes.append(rutaCompleta)
        
        random.shuffle(self.imagenes)

        if self.imagenes:
            self.indiceActual = 0
            self.mostrarImagen()

    def predecirImagen(self):

        rutaImagen = self.imagenes[self.indiceActual]
        carpetaPadre = os.path.basename(os.path.dirname(rutaImagen))

        imagen = cv2.imread(rutaImagen)
        imagenRedimensionada = cv2.resize(imagen, (227, 227))
        imagenAlexnet = cv2.cvtColor(imagenRedimensionada, cv2.COLOR_BGR2RGB)
        imagenAlexnet = np.expand_dims(imagenAlexnet, axis=0) 

        if(carpetaPadre == "docks"):

            self.valorEsperado.configure(text = "Plaga")
            carpetaPadre = "Plaga"

        else:
            
            self.valorEsperado.configure(text = "No Plaga")
            carpetaPadre = "No Plaga"

        porcentaje = self.modelo.predict(imagenAlexnet, verbose = 0)
        porcentaje = porcentaje[0]
        prediccion = None
        valor = None

        if(porcentaje > 0.5): # notdocks
            prediccion = "No Plaga"
        else: # docks
            prediccion = "Plaga"
            porcentaje = 1 - porcentaje 

        self.valorPrediccion.configure(text = f"{prediccion}")
        self.valorPorcentaje.configure(text = f"{porcentaje * 100} %")     

        if(prediccion == carpetaPadre):
            self.valorPrediccion.configure(text_color = self.verde)
            self.valorPorcentaje.configure(text_color = self.verde)    
        else:
            self.valorPrediccion.configure(text_color = "red") 
            self.valorPorcentaje.configure(text_color = "red")    

        print(porcentaje)

    def mostrarImagen(self):

        if self.indiceActual < len(self.imagenes):
    
            rutaImagen = self.imagenes[self.indiceActual]
            imagen = Image.open(rutaImagen)

            anchoFrame = self.imagenesAnchoFrame
            largoFrame = self.imagenesLargoFrame - 40

            anchoImagen, largoImagen = imagen.size

            anchoRescalado = anchoFrame / anchoImagen 
            largoRescalado = largoFrame / largoImagen
            rescalado = min(anchoRescalado, largoRescalado)

            ancho = int(anchoImagen * rescalado)
            largo = int(largoImagen * rescalado)

            imagen = imagen.resize((ancho, largo))
            imagen_ctk = CTkImage(imagen, size=(ancho, largo))

            contenedorImagen = CTkLabel(master=self.imagenesFrame, text="", image=imagen_ctk)
            contenedorImagen.place(relx=0.5, rely=0.5, anchor="center")

            self.indiceActual += 1

            self.predecirImagen()

            self.app.after(3000, self.mostrarImagen)

if __name__ == "__main__":

    Ventana()