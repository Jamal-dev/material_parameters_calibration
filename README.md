# Material Parameter Calibration

This project is dedicated to the calibration of material parameters based on the constitutive model defined by the strain energy function, $\Psi$. It specifically caters to hyperelastic materials, accommodating anisotropic behaviors represented through fiber directions. Currently, the model assumes a single fiber orientation, primarily aligned along the x-axis. This tool is invaluable for material scientists and engineers looking to calibrate and validate hyperelastic material models under various loading conditions.

## Features

- Supports six distinct states of stress for comprehensive material analysis.
- Processes input data from CSV files detailing different types of material loading scenarios.
- Incorporates material anisotropy by allowing the specification of fiber direction.
- Utilizes a customizable mathematical description for $\Psi$, accommodating a range of hyperelastic models.

  ![gui_layout]([data/gui_layout.png](https://github.com/Jamal-dev/material_parameters_calibration/blob/main/data/gui_layout.png))

## Mathematical Model

The function $\Psi$, representing the strain energy potential, is defined as:
```
"c1 * (J-1)**2 + c2 * (I1bar-3) + c3 * (I1bar-3)**2 + c4 * (I2bar-3) + c5 * (I2bar-3)**2 + c6 * (I4bar-1) + c7 * (I5bar-1)"
```
where $c1, c2,..., c7$ are the material parameters to be calibrated, and $I1bar, ..., I5bar$ are the invariants related to the deformation state.

## Data Format

The input CSV file should contain stress-strain data structured as follows:
- The first six columns should represent strain components: $\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33}, \varepsilon_{12}, \varepsilon_{13}, \varepsilon_{23}$.
- The following six columns should correspond to stress components: $\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}$.
- The 13th column specifies the type of loading (1 to 6), delineating uniaxial to pure shear loadings in different orientations.

## Installation

Ensure you have Python and necessary dependencies installed. Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/Jamal-dev/material_parameters_calibaration/
cd material_parameters_calibaration
pip install -r requirements.txt
```

## Usage

Launch the calibration tool from the command line:

```bash
python gui_layout.py
```

Within the graphical user interface:
1. Specify the path to your input CSV file.
2. Enter the mathematical expression for $\Psi$.
3. Set the calibration parameters, such as fiber direction and batch size.
4. Define the output settings like `model_data.title` and `model_data.folder_name`.
5. Click "Train Start!" to initiate the calibration process.

Results, including the calibrated model and plots, will be displayed within the GUI and saved to the specified directory.

## Contributions

Feedback, issues, and pull requests are welcome to improve this project and extend its capabilities.

## Contact

For questions or support, contact Jamal Ahmed at jamalahmed68@gmail.com.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
