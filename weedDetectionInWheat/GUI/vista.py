from customtkinter import *
from PIL import Image

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
        self.fondoAzul = "#1565C0"

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

        self.generarImagenes(imagenes, resultados, imagenesAncho, imagenesLargo, resultadosAncho, resultadosLargo)

    def generarImagenes(self, primerFrame, segundoFrame, primerFrameAncho, primerFrameLargo, segundoFrameAncho, segundoFrameLargo):

        pass

if __name__ == "__main__":

    Ventana()