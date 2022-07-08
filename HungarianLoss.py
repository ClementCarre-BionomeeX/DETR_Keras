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
            * <b, N, 4> "bbox tensor"
            * <b, N, C + 1> "classe probs tensor"
        - ytrue:
            * <b, N, 4> "bbox tensor"
            * <b, N, C + 1> "classe probs tensor"

        b = batchsize
        N = number of queries:
            For targets, we fill the missing values with random boxes with class 0
        C = number of classes:
            We add a zero-th class : no class

    """
    indices = tf.stop_gradient(_hungarian_matching(ypred, ytrue, cost_class, cost_bbox, cost_giou))


    pass

def _loss_label()


def _hungarian_matching(
    outputs,
    targets,
    cost_class: float = 1,
    cost_bbox: float = 1,
    cost_giou: float = 1,
):
    pass
