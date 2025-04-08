from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from bdgs.algorithms.maung.maung import Maung
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.data.algorithm import ALGORITHM

ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.MAUNG: Maung(),
    ALGORITHM.ADITHYA_RAJESH: AdithyaRajesh(),
}
