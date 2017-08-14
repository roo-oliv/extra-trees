import numpy


def filter_out_constants(attributes: numpy.ndarray) -> numpy.ndarray:
    if attributes.size == 0:
        return attributes
    transposed = numpy.transpose(attributes, (1, 0))
    for values, i in zip(transposed, range(len(transposed))):
        if numpy.unique(values).size == 1:
            transposed = numpy.delete(transposed, i, 0)
    return numpy.transpose(transposed, (1, 0))
