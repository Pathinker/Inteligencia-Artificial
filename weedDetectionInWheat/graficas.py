import matplotlib.pyplot as plt

primerAciertoEntrenamiento = [0.7070, 0.7724, 0.8059, 0.8272, 0.8319, 0.8450, 0.8465, 0.8658, 0.8590, 0.8621, 0.8685, 0.8790, 0.8918, 0.8871, 0.8872, 0.8910, 0.9037, 0.9071, 0.9108, 0.9168, 0.9140, 0.9293, 0.9308, 0.9363, 0.9313, 0.9426, 0.9433, 0.9489, 0.9542, 0.9570, 0.9617, 0.9639, 0.9603, 0.9732, 0.9697, 0.9715, 0.9760, 0.9779, 0.9745, 0.9823, 0.9807, 0.9852, 0.9896, 0.9870, 0.9881, 0.9869, 0.9858, 0.9881, 0.9884, 0.9925]
primerPerdidaEntrenamiento = [0.7887, 0.5785, 0.4991, 0.4518, 0.4360, 0.4135, 0.3945, 0.3583, 0.3647, 0.3464, 0.3331, 0.3096, 0.2836, 0.2958, 0.2876, 0.2661, 0.2565, 0.2357, 0.2375, 0.2178, 0.2202, 0.1941, 0.1862, 0.1722, 0.1723, 0.1561, 0.1524, 0.1442, 0.1301, 0.1179, 0.1088, 0.1002, 0.0969, 0.0822, 0.0871, 0.0805, 0.0723, 0.0648, 0.0808, 0.0525, 0.0540, 0.0462, 0.0359, 0.0426, 0.0379, 0.0388, 0.0375, 0.0431, 0.0329, 0.0250]
primerAciertoValidacion = [0.4493, 0.8104, 0.8119, 0.8299, 0.8731, 0.8269, 0.8776, 0.8761, 0.8836, 0.8701, 0.8791, 0.8896, 0.8881, 0.8866, 0.8701, 0.9000, 0.8836, 0.8955, 0.8687, 0.8776, 0.8940, 0.8597, 0.9045, 0.9030, 0.8940, 0.9075, 0.9015, 0.8806, 0.9254, 0.8955, 0.7746, 0.9134, 0.9015, 0.8776, 0.9239, 0.9119, 0.9030, 0.8090, 0.9104, 0.8776, 0.8881, 0.9075, 0.8687, 0.9030, 0.8881, 0.8970, 0.8836, 0.9104, 0.8836, 0.9045]
primerPerdidaValidacion = [0.7718, 0.4443, 0.4352, 0.4107, 0.3402, 0.4245, 0.3506, 0.3321, 0.3061, 0.3416, 0.3160, 0.2912, 0.2907, 0.3077, 0.3649, 0.2807, 0.3507, 0.2879, 0.4035, 0.3803, 0.2914, 0.5472, 0.2745, 0.2940, 0.3333, 0.2806, 0.2755, 0.4460, 0.2746, 0.3612, 0.4871, 0.2981, 0.4093, 0.4792, 0.2778, 0.3407, 0.3186, 1.4196, 0.3275, 0.3822, 0.5217, 0.3962, 0.6441, 0.4328, 0.5382, 0.5696, 0.4228, 0.3685, 0.5860, 0.4647]

segundoAciertoEntrenamiento = [0.5618, 0.6043, 0.7326, 0.7166, 0.7910, 0.7616, 0.7567, 0.7950, 0.7819, 0.8240, 0.7844, 0.8317, 0.8042, 0.8367, 0.8356, 0.8113, 0.8341, 0.8511, 0.8631, 0.8556, 0.8737, 0.8571, 0.8787, 0.8525, 0.8822, 0.8776, 0.8715, 0.8607, 0.8393, 0.8865, 0.8963, 0.8930, 0.8931, 0.8961, 0.8889, 0.8870, 0.8904, 0.8971, 0.8991, 0.8869, 0.8790, 0.8934, 0.8962, 0.9018, 0.9064, 0.8964, 0.9036, 0.8979, 0.8996, 0.9043]
segundoPerdidaEntrenamiento = [6.1537, 0.7675, 0.5843, 0.5780, 0.5331, 0.5992, 0.5953, 0.5649, 0.6010, 0.5371, 0.5453, 0.4996, 0.5469, 0.5029, 0.5481, 0.6113, 0.5198, 0.4797, 0.4629, 0.4599, 0.4165, 0.4723, 0.4480, 0.5618, 0.4297, 0.4454, 0.4331, 0.4844, 0.5465, 0.4117,0.3962, 0.4074, 0.3987, 0.3890, 0.3958, 0.4009, 0.4827, 0.3598, 0.4062, 0.4195, 0.4073, 0.3741, 0.3806, 0.3529, 0.3427, 0.3513, 0.3525, 0.3636, 0.3603, 0.3518]
segundoAciertoValidacion = [0.4284, 0.8313, 0.7388, 0.8537, 0.2164, 0.8358, 0.8746, 0.7791, 0.8806, 0.8075, 0.8687, 0.8075, 0.8090, 0.8597, 0.8030, 0.8985, 0.7418, 0.9030, 0.4910, 0.9015, 0.8343, 0.8642, 0.8761, 0.8821, 0.9149, 0.9134, 0.8851, 0.8284, 0.9224, 0.9269, 0.9045, 0.9179, 0.9104, 0.8761, 0.9239, 0.8134, 0.9179, 0.9194, 0.9239, 0.9179, 0.8687, 0.9328, 0.9239, 0.8970, 0.9015, 0.8597, 0.9343, 0.8507, 0.9418, 0.9224]
segundoPerdidaValidacion = [0.8142, 0.4420, 0.5442, 0.3852, 1.5360, 0.5297, 0.5723, 0.5049, 0.5400, 0.4935, 0.5247, 0.5537, 0.5471, 0.3975, 0.5535, 0.4394, 0.5870, 0.4047, 1.4650, 0.3969, 0.3561, 0.4807, 0.4145, 0.4079, 0.3638, 0.3543, 0.3920, 0.5391, 0.3391, 0.3372, 0.2751, 0.3481, 0.2465, 0.4073, 0.3051, 1.0882, 0.3243, 0.2721, 0.3034, 0.2592, 0.3576, 0.2852, 0.2802, 0.3038, 0.3153, 0.4529, 0.2487, 0.4004, 0.2441, 0.3272]

tercerAciertoEntrenamiento = [0.5593, 0.6565, 0.6348, 0.6976, 0.7399, 0.6756, 0.7430, 0.7565, 0.7972, 0.8066, 0.7790, 0.6925, 0.8145, 0.7796, 0.7679, 0.7400, 0.8023, 0.8090, 0.7876, 0.7967, 0.8164, 0.8029, 0.7777, 0.8146, 0.7889, 0.8394, 0.8153, 0.8348, 0.8302, 0.8385, 0.7637, 0.7344, 0.7863, 0.8243, 0.8488, 0.8559, 0.8697, 0.8751, 0.8703, 0.8698, 0.8794, 0.8589, 0.8628, 0.8660, 0.8526, 0.8496, 0.8489, 0.8472, 0.8702, 0.8509]
tercerPerdidaEntrenamiento = [8.7163, 0.6381, 0.6513, 0.6104, 0.5656, 0.6137, 0.6215, 0.6082, 0.5184, 0.5274, 0.5382, 0.6109, 0.5361, 0.5731, 0.6906, 0.5897, 0.5006, 0.4999, 0.5470, 0.5238, 0.4970, 0.5576, 0.6231, 0.4937, 0.5583, 0.5075, 0.5476, 0.5172, 0.4933, 0.4977, 0.6941, 0.6396, 0.5987, 0.5061, 0.4922, 0.4985, 0.4575, 0.4752, 0.4739, 0.4506, 0.4372, 0.4626, 0.5070, 0.4678, 0.4913, 0.4869, 0.4908, 0.4561, 0.4319, 0.5566]
tercerAciertoValidacion = [0.1940, 0.3299, 0.8045, 0.5612, 0.4194, 0.5373, 0.3687, 0.8209, 0.6866, 0.7761, 0.5030, 0.8851, 0.3284, 0.8060, 0.8149, 0.3090, 0.5537, 0.5910, 0.7478, 0.2388, 0.7224, 0.8149, 0.9015, 0.5463, 0.8612, 0.8343, 0.7821, 0.8597, 0.8493, 0.8851, 0.7985, 0.7373, 0.8478, 0.8940, 0.9060, 0.9090, 0.9000, 0.8955, 0.7448, 0.8104, 0.8776, 0.8164, 0.8060, 0.8343, 0.5552, 0.7896, 0.8090, 0.6149, 0.8896, 0.8313]
tercerPerdidaValidacion = [4.5973, 1.0523, 0.5997, 0.7714, 0.9592, 0.6615, 1.0174, 0.5638, 0.7221, 0.5289, 0.7268, 0.5930, 1.2645, 0.4942, 0.5361, 1.2444, 1.0322, 0.8658, 0.5935, 1.2754, 0.6677, 0.5101, 0.4432, 0.7954, 0.5706, 0.5368, 0.6085, 0.5200, 0.5256, 0.5553, 0.5609, 0.8444, 0.5294, 0.4598, 0.4414, 0.4474, 0.4386, 0.4607, 0.7319, 0.5116, 0.4464, 0.4992, 0.5097, 0.4807, 0.8829, 0.4942, 0.5024, 1.2254, 0.5055, 0.4686]

cuartoAciertoEntrenamiento = [0.8031, 0.7993, 0.8004, 0.8017, 0.8015, 0.8017, 0.7999, 0.1992, 0.7989, 0.8007, 0.8002, 0.8013, 0.7979, 0.8002, 0.7999, 0.8000, 0.7983, 0.1975, 0.7992, 0.1995, 0.1977, 0.8004, 0.8004, 0.7988, 0.7997, 0.2008, 0.7990, 0.8017, 0.2360, 0.1979, 0.7990, 0.8016, 0.7995, 0.2014, 0.8015, 0.1990, 0.8011, 0.8024, 0.8034, 0.8010, 0.8000, 0.8016, 0.8003, 0.8018, 0.8002, 0.8008, 0.8010, 0.8001, 0.7999, 0.8015]
cuartoPerdidaEntrenamiento = 5814774218601087006600232697856.0, 157803.9375, 1215572.5, 17078248.0, float('nan'), 6822426112.0, 158569904.0, 0.7770, 300614413318155914381885440.0, 37679.8711, 59797.0391, 282059734245818083796175552512.0, 1867218267160469451664112222208.0, 24577367616551481180700930998272.0, 66773.1328, 759861.0625, 8724260.0, 0.8241, 929057.875, 0.7732, 8.0780, 19453.1133, 8588.1406, 33.3693, 3393.6958, 0.8107, 31270.4199, 29.6042, float('nan'), 11.3600,1197.7317, 0.6477, 1279.8802, 0.8291, 5519.6675, 0.8491, 10758.6172, 6981.52, 0.5188, 155.7539, 149.0787, 453.4644, 48.2255, 123.9781, 86.0982, 172.5682, 634.4149, 220.6361, 43.7756, 32.1709
cuartoAciertoValidacion = [0.8049, 0.8215, 0.8080, 0.8173, 0.8015, 0.8180, 0.7996, 0.1744, 0.8271, 0.8215, 0.8312, 0.8322, 0.8175, 0.7900, 0.8132, 0.8069, 0.8018, 0.1801, 0.8040, 0.1795, 0.1828, 0.7934, 0.8160, 0.8099, 0.8221, 0.1891, 0.8165, 0.8076, 0.2267, 0.1923, 0.8019, 0.8192, 0.8201, 0.1976, 0.8057, 0.1804, 0.8053, 0.8053, 0.8223, 0.8275, 0.8129, 0.8258, 0.8250, 0.8171, 0.8110, 0.8241, 0.8109, 0.8058, 0.8228, 0.8003]
cuartoPerdidaValidacion = [5885750857933571692761819643904.0, 140365.25, 1169279.25, 15737332.0, float('nan'), 6261828096.0, 158834240.0, 0.7832, 263748189458477753867173888.0, 33746.9023, 50521.5703, 236723070395278547450865909760.0, 1685920044348959617364185317376.0, 26481116297434700785794433417216.0, 62319.7227, 733638.1875, 8573746.0, 0.8306, 906751.0625, 0.7780, 8.2279, 20132.9219, 7916.4893, 31.5274, 3014.8071, 0.8148, 28549.9434, 28.7221, float('nan'), 11.4386, 1180.5522, 0.5942, 1148.2313, 0.8306, 5401.9902, 0.8572, 10527.9990, 6879.1274, 0.4813, 134.9702, 139.4873, 398.0623, 42.2771, 114.4083, 81.4494, 152.4557, 602.9380, 214.4249, 38.7674, 32.3638]

epochs = []

for i in range(50):

    epochs.append(i + 1)

def graficar(aciertoEntrenamiento, aciertoValidacion, perdidaEntrenamiento, perdidaValidacion):

    plt.title("Precisión Modelo")
    plt.plot(epochs, aciertoEntrenamiento, label = "Entrenamiento", color = "blue")
    plt.plot(epochs, aciertoValidacion, label = "Validación", color = "red")
    #plt.axhline(y = max(aciertoEntrenamiento), linestyle = "--", color = "darkblue")
    #plt.axhline(y = max(aciertoValidacion), linestyle = "--", color = "darkred")
    plt.xlabel("Epochs")
    plt.ylabel("Porcentaje")
    plt.legend()
    plt.show()

    print(max(aciertoEntrenamiento))
    print(aciertoEntrenamiento.index(max(aciertoEntrenamiento)))
    print(max(aciertoValidacion))
    print(aciertoValidacion.index(max(aciertoValidacion)))

    plt.title("Error Modelo")
    plt.plot(epochs, perdidaEntrenamiento, label = "Entrenamiento", color = "blue")
    plt.plot(epochs, perdidaValidacion, label = "Validación", color = "red")
    #plt.axhline(y = min(perdidaEntrenamiento), linestyle = "--", color = "darkblue")
    #plt.axhline(y = min(perdidaValidacion), linestyle = "--", color = "darkred")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    print(min(perdidaEntrenamiento))
    print(perdidaEntrenamiento.index(min(perdidaEntrenamiento)))
    print(min(perdidaValidacion))
    print(perdidaValidacion.index(min(perdidaValidacion)))

graficar(cuartoAciertoEntrenamiento, cuartoAciertoValidacion, cuartoPerdidaEntrenamiento, cuartoPerdidaValidacion)