import tensorflow as tf

def hungarianLoss(
    ytrue,
    ypred,
    cost_class: float = 1,
    cost_bbox: float = 1,
    cost_giou: float = 1,
):
    truebbox, trueclasses = ytrue
    predbbox, predclasses = ypred

    def curried_hungarian_loss(ytrue,
    ypred,
    cost_class: float = 1,
    cost_bbox: float = 1,
    cost_giou: float = 1,):
        return _hungarianLoss()




def _hungarianLoss(
    ytrue,
    ypred,
    cost_class: float = 1,
    cost_bbox: float = 1,
    cost_giou: float = 1,
):

    """Hungarian matching Loss function

    Params:
        - ypred:
            * <b, N, 4 "bbox tensor" + C + 1 "classe probs tensor">
        - ytrue:
            * <b, N, 4 "bbox tensor" + C + 1 "classe probs tensor">

        b = batchsize
        N = number of queries:
            For targets, we fill the missing values with 0 sized boxes with class 0
        C = number of classes:
            We add a zero-th class : the no-object class

    """
    pass


def loss_label(ytrue, ypred, indices, num_boxes):

    indices = tf.stop_gradient(_hungarian_matcher(ytrue, ypred))



def _hungarian_matcher_on_batch(
    targets,
    outputs,
):
    """
    targets :
        <N, 4 "bbox tensor" + C + 1 "classe probs tensor">
    outputs :
        <N, 4 "bbox tensor" + C + 1 "classe probs tensor">
    """
    ta_bbox = targets[:, :4]
    ta_clss = targets[:, 4:]
    ou_bbox = outputs[:, :4]
    ou_clss = outputs[:, 4:]

    # We want to construct a cost matrix of size N, N
