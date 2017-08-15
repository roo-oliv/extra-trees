import numpy


def filter_out_constants(
        attributes: numpy.ndarray, sample: list) -> numpy.ndarray:
    if attributes.size == 0:
        return attributes
    transposed = numpy.transpose(attributes, (1, 0))
    delete = []
    for values, i in zip(transposed, range(len(transposed))):
        if numpy.unique(values).size == 1:
            delete.append(i)

    transposed = numpy.delete(transposed, delete, 0)
    for i in sorted(delete, reverse=True):
        del sample[i]
    return numpy.transpose(transposed, (1, 0))
