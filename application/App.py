import monai.deploy.core as md  # 'md' stands for MONAI Deploy (or can use 'core' instead)
from monai.deploy.core import (
    Application,
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)


@md.resource(cpu=1)
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
@md.env(pip_packages=["scikit-image >= 0.17.2"])
class App(Application):
    """This is a very basic application.

    This showcases the MONAI Deploy application framework.
    """

    # App's name. <class name>('App') if not specified.
    name = "simple_imaging_app"
    # App's description. <class docstring> if not specified.
    description = "This is a very simple application."
    # App's version. <git version tag> or '0.0.0' if not specified.
    version = "0.1.0"

    def compose(self):
        """This application has three operators.

        Each operator has a single input and a single output port.
        Each operator performs some kind of image processing function.
        """
        sobel_op = SobelOperator()
        median_op = MedianOperator()
        gaussian_op = GaussianOperator()

        self.add_flow(sobel_op, median_op)
        # self.add_flow(sobel_op, median_op, {"image": "image"})
        # self.add_flow(sobel_op, median_op, {"image": {"image"}})

        self.add_flow(median_op, gaussian_op)


@md.input("image", DataPath, IOType.DISK)
@md.output("image", Image, IOType.IN_MEMORY)
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class SobelOperator(Operator):
    """This Operator implements a Sobel edge detector.

    It has a single input and single output.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage import filters, io

        input_path = op_input.get().path
        if input_path.is_dir():
            input_path = next(input_path.glob("*.*"))  # take the first file

        data_in = io.imread(input_path)[:, :, :3]  # discard alpha channel if exists
        data_out = filters.sobel(data_in)

        op_output.set(Image(data_out))


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("image", Image, IOType.IN_MEMORY)
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class MedianOperator(Operator):
    """This Operator implements a noise reduction.

    The algorithm is based on the median operator.
    It ingests a single input and provides a single output.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage.filters import median

        data_in = op_input.get().asnumpy()
        data_out = median(data_in)
        op_output.set(Image(data_out))


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("image", DataPath, IOType.DISK)
# If `pip_packages` is specified, the definition will be aggregated with the package dependency list of other
# operators and the application in packaging time.
# @md.env(pip_packages=["scikit-image >= 0.17.2"])
class GaussianOperator(Operator):
    """This Operator implements a smoothening based on Gaussian.

    It ingests a single input and provides a single output.
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        from skimage.filters import gaussian
        from skimage.io import imsave

        data_in = op_input.get().asnumpy()
        data_out = gaussian(data_in, sigma=0.2)

        output_folder = op_output.get().path
        output_path = output_folder / "final_output.png"
        imsave(output_path, data_out)


# Run the application when this file is executed.
if __name__ == "__main__":
    App(do_run=True)
