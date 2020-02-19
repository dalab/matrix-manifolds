from graphembed.train import TrainingEngine as Base


class TrainingEngine(Base):

    def __init__(self, *args, **kwargs):
        if 'stabilize_every_epochs' in kwargs and \
                kwargs['stabilize_every_epochs'] > 1:
            raise ValueError('For product-space training using the Universal '
                             'manifold we need to stabilize every epoch in '
                             'order to impose the norm constraint.')
        super().__init__(*args, **kwargs)
        self.stabilize_every_epochs = 1

    def _burnin(self, graph_dataset):
        # The actual burnin epochs.
        for epoch in range(1, self.burnin_epochs):
            _ = self._train(graph_dataset, self.alpha, epoch)
            self.emb.stabilize()
