

class DecisionTree:

    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.threshold = None
        self.feature = None
        self.prediction = None
        self.is_leaf = None

    def insert(self, threshold, feature, prediction, is_leaf):
        self.threshold = threshold
        self.prediction = prediction
        self.feature = feature
        self.is_leaf = is_leaf

    # def printTree(self):
    #     if self.left:
    #         self.left.printTree()
    #     print("Feature : ", self.feature, " | Threshold : ", self.threshold)
    #     if self.prediction is not None:
    #         print("Prediction : ", self.prediction)
    #     if self.right:
    #         self.right.printTree()

    def print_tree(self, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if self.is_leaf:
            print(spacing + "Predict", self.prediction)
            return

        print("Feature : ", self.feature, " Threshold : ", self.threshold)
        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.left.print_tree()
        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.right.print_tree()
