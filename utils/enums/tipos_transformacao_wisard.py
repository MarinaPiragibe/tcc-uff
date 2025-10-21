from enum import Enum


class TiposDeTransformacao(Enum):
    BASICA = "basica"
    ESCALA_DE_CINZA = "escala_de_cinza"
    THRESHOLD_3 = "threshold_3"
    THRESHOLD_31 = "threshold_31"
    STRIDE_HD = "stride_hd"