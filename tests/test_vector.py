from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class TestVector:
    def test_replace_vector_part_bit_index0(self):
        vec_len = 5
        v1 = BitStringVector(SimpleFitness(), length=vec_len)
        v1.vector = [0] * vec_len

        v2 = BitStringVector(SimpleFitness(), length=vec_len)
        v2.vector = [1] * vec_len

        replaced_part_size = 3
        old_v1_part = v1.replace_vector_part(v2.vector[:replaced_part_size], 0)
        assert v1.vector == [1] * replaced_part_size + [0] * (vec_len - replaced_part_size)

        old_v2_part = v2.replace_vector_part(old_v1_part, 0)
        assert old_v2_part == [1] * replaced_part_size
        assert v2.vector == [0] * replaced_part_size + [1] * (vec_len - replaced_part_size)

    def test_replace_vector_part_bit_middle_index(self):
        v1 = BitStringVector(SimpleFitness(), length=4)
        v1.vector = [0, 1] * 2

        v2 = BitStringVector(SimpleFitness(), length=4)
        v2.vector = [1, 0] * 2

        old_v1_part = v1.replace_vector_part(v2.vector[1:3], 1)
        assert v1.vector == [0, 0, 1, 1]
        assert old_v1_part == [1, 0]

        old_v2_part = v2.replace_vector_part(old_v1_part, 1)
        assert old_v2_part == [0, 1]
        assert v2.vector == [1, 1, 0, 0]
