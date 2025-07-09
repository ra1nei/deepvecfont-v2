import random
import numpy as np
from PIL import Image, ImageDraw
from deap import base, creator, tools, algorithms
import math
import torch
import os

# --- Placeholder for your actual opts and util_funcs ---
# IMPORTANT: You MUST replace these with your actual opts and util_funcs
# imported from your project (e.g., from your 'options.py' and 'models.util_funcs').

class MockOpts:
    def __init__(self, img_size=64, max_seq_len=50, dim_seq_short=9):
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.dim_seq_short = dim_seq_short # 1 command type + 8 arguments

class MockUtilFuncs:
    def __init__(self, n_quant_bins=128, max_coord_val=30.0):
        self.N_QUANT_BINS = n_quant_bins
        self.MAX_COORD_VAL = max_coord_val

    def numericalize(self, coord_tensor):
        if not isinstance(coord_tensor, torch.Tensor):
            coord_tensor = torch.tensor(coord_tensor, dtype=torch.float32)
        clipped_tensor = torch.clamp(coord_tensor, 0.0, self.MAX_COORD_VAL)
        numericalized_tensor = torch.round(clipped_tensor / self.MAX_COORD_VAL * (self.N_QUANT_BINS - 1))
        return numericalized_tensor.long()

    def denumericalize(self, numerical_tensor):
        if not isinstance(numerical_tensor, torch.Tensor):
            numerical_tensor = torch.tensor(numerical_tensor, dtype=torch.float32)
        denumericalized_tensor = numerical_tensor / (self.N_QUANT_BINS - 1) * self.MAX_COORD_VAL
        return denumericalized_tensor

# Default options for the evolver, can be overridden
_default_evolver_opts = MockOpts()
_default_util_funcs = MockUtilFuncs()

# --- DEAP setup (needs to be done once) ---
# Create Fitness and Individual types globally so they are defined only once
# when the module is imported.
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
except RuntimeError:
    # Types already created, ignore (happens if module is reloaded in interactive env)
    pass


class SVGEvolver:
    def __init__(self, opts=_default_evolver_opts, util_funcs=_default_util_funcs):
        self.opts = opts
        self.util_funcs = util_funcs
        
        self.ARGS_PER_COMMAND = self.opts.dim_seq_short - 1 
        self.NUM_SVG_COMMANDS_IN_INDIVIDUAL = self.opts.max_seq_len
        self.INDIVIDUAL_SIZE = self.NUM_SVG_COMMANDS_IN_INDIVIDUAL * self.opts.dim_seq_short

        self.toolbox = base.Toolbox()
        self._register_toolbox_operators()

        self.target_image_np = None # Will be set by set_target
        self.image_size = (self.opts.img_size, self.opts.img_size)

    def _register_toolbox_operators(self):
        self.toolbox.register("attr_gene", random.random) # Gene values are floats [0,1]
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                              self.toolbox.attr_gene, n=self.INDIVIDUAL_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # The evaluate function needs target_image_np, which is set dynamically
        self.toolbox.register("evaluate", self._evaluate_svg_individual, 
                              max_coord_val=self.util_funcs.MAX_COORD_VAL, 
                              n_quant_bins=self.util_funcs.N_QUANT_BINS)

        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.05)

    def set_target(self, target_image_np, initial_sequence_np=None):
        """
        Sets the target image for the evolutionary process and optionally
        provides an initial SVG sequence for population seeding.

        Args:
            target_image_np (np.array): The target raster image (HxW, grayscale, 0-1 float).
            initial_sequence_np (np.array, optional): A numericalized SVG sequence (max_seq_len, dim_seq_short)
                                                      to seed the initial population.
        """
        if target_image_np.ndim == 3 and target_image_np.shape[0] == 1:
            target_image_np = target_image_np.squeeze(0) # Remove batch dim
        if target_image_np.ndim == 3 and target_image_np.shape[-1] == 1:
            target_image_np = target_image_np.squeeze(-1) # Remove channel dim if 1
        
        if target_image_np.max() > 1.0: # Normalize if 0-255 range
            target_image_np = target_image_np / 255.0
        
        self.target_image_np = target_image_np
        self.image_size = (target_image_np.shape[0], target_image_np.shape[1])
        
        # Pass the target image to the evaluate function through the toolbox arguments
        self.toolbox.evaluate.keywords['target_image_np'] = self.target_image_np
        self.toolbox.evaluate.keywords['image_size'] = self.image_size

        self.initial_sequence_ga_format = None
        if initial_sequence_np is not None:
            # Ensure initial_sequence_np is (max_seq_len, dim_seq_short)
            if initial_sequence_np.ndim == 3: # If (batch, max_seq, dim_seq_short)
                initial_sequence_np = initial_sequence_np[0] # Take first item in batch

            flat_sequence_ga_format = []
            for cmd_vec in initial_sequence_np:
                cmd_type_numerical = cmd_vec[0]
                args_numerical = cmd_vec[1:].tolist() # Ensure it's a list

                cmd_type_raw_ga = cmd_type_numerical / 3.0 # Scale command type back to [0,1]
                # Scale arguments from numerical bins back to [0,1] for GA's internal representation
                args_raw_ga = [float(a_num) / (self.util_funcs.N_QUANT_BINS - 1) for a_num in args_numerical] 
                
                flat_sequence_ga_format.extend([cmd_type_raw_ga] + args_raw_ga)
            
            # Pad or truncate to match INDIVIDUAL_SIZE
            if len(flat_sequence_ga_format) > self.INDIVIDUAL_SIZE:
                flat_sequence_ga_format = flat_sequence_ga_format[:self.INDIVIDUAL_SIZE]
            elif len(flat_sequence_ga_format) < self.INDIVIDUAL_SIZE:
                flat_sequence_ga_format.extend([0.0] * (self.INDIVIDUAL_SIZE - len(flat_sequence_ga_format)))
            
            self.initial_sequence_ga_format = flat_sequence_ga_format


    def _draw_svg_commands_on_image(self, individual, image_size, max_coord_val, n_quant_bins):
        """
        Simulates drawing SVG commands based on the individual's values.
        This is a simplified rasterizer using Pillow.
        (Same logic as before, but now within the class)
        """
        img = Image.new('L', image_size, color='black')
        draw = ImageDraw.Draw(img)
        current_x, current_y = 0.0, 0.0

        for i in range(self.NUM_SVG_COMMANDS_IN_INDIVIDUAL):
            start_idx = i * self.opts.dim_seq_short
            if start_idx + self.opts.dim_seq_short > len(individual):
                break

            cmd_type_raw = individual[start_idx]
            cmd_type = int(round(cmd_type_raw * 3))
            cmd_type = max(0, min(3, cmd_type))

            args_raw_list = individual[start_idx + 1 : start_idx + 1 + self.ARGS_PER_COMMAND]
            
            numerical_args_mock = [int(round(a_raw * (n_quant_bins - 1))) for a_raw in args_raw_list]
            args_real_coords_tensor = self.util_funcs.denumericalize(numerical_args_mock) # Use util_funcs.denumericalize
            args_real_coords = args_real_coords_tensor.tolist()

            pixel_coords = []
            for j in range(0, len(args_real_coords), 2):
                px = int(args_real_coords[j] / max_coord_val * image_size[0])
                py = int(args_real_coords[j+1] / max_coord_val * image_size[1])
                pixel_coords.extend([px, py])
            pixel_coords = [max(0, min(image_size[0]-1, c)) if idx % 2 == 0 else max(0, min(image_size[1]-1, c)) for idx, c in enumerate(pixel_coords)]

            if cmd_type == 0: # MoveTo
                if len(pixel_coords) >= 2:
                    current_x, current_y = pixel_coords[0], pixel_coords[1]
                    draw.ellipse((current_x-1, current_y-1, current_x+1, current_y+1), fill='white', outline='white')
            elif cmd_type == 1: # LineTo
                if len(pixel_coords) >= 2:
                    x_to, y_to = pixel_coords[0], pixel_coords[1]
                    draw.line((current_x, current_y, x_to, y_to), fill='white', width=1)
                    current_x, current_y = x_to, y_to
            elif cmd_type == 2: # Quadratic Bezier
                if len(pixel_coords) >= 4:
                    p0x, p0y = current_x, current_y 
                    p1x, p1y = pixel_coords[0], pixel_coords[1]
                    p2x, p2y = pixel_coords[2], pixel_coords[3]
                    num_segments = 20
                    for t_idx in range(num_segments + 1):
                        t = t_idx / num_segments
                        x = (1 - t)**2 * p0x + 2 * (1 - t) * t * p1x + t**2 * p2x
                        y = (1 - t)**2 * p0y + 2 * (1 - t) * t * p1y + t**2 * p2y
                        if t_idx > 0: draw.line((prev_x, prev_y, x, y), fill='white', width=1)
                        prev_x, prev_y = x, y
                    current_x, current_y = p2x, p2y 
            elif cmd_type == 3: # Cubic Bezier
                if len(pixel_coords) >= 6: 
                    p0x, p0y = current_x, current_y 
                    p1x, p1y = pixel_coords[0], pixel_coords[1]
                    p2x, p2y = pixel_coords[2], pixel_coords[3]
                    p3x, p3y = pixel_coords[4], pixel_coords[5]
                    num_segments = 20
                    for t_idx in range(num_segments + 1):
                        t = t_idx / num_segments
                        x = (1-t)**3 * p0x + 3*(1-t)**2 * t * p1x + 3*(1-t) * t**2 * p2x + t**3 * p3x
                        y = (1-t)**3 * p0y + 3*(1-t)**2 * t * p1y + 3*(1-t) * t**2 * p2y + t**3 * p3y
                        if t_idx > 0: draw.line((prev_x, prev_y, x, y), fill='white', width=1)
                        prev_x, prev_y = x, y
                    current_x, current_y = p3x, p3y 
        return np.array(img) / 255.0

    def _evaluate_svg_individual(self, individual, target_image_np, image_size, max_coord_val, n_quant_bins):
        """Fitness function called by DEAP."""
        generated_image_np = self._draw_svg_commands_on_image(individual, image_size, max_coord_val, n_quant_bins)
        mse = np.mean((generated_image_np - target_image_np)**2)
        return (mse,)

    def evolve_svg_for_image(self, population_size=200, generations=1000, cxpb=0.7, mutpb=0.3, 
                             seed_ratio=0.1, verbose=True):
        """
        Runs the evolutionary algorithm to find an SVG sequence for the set target image.

        Args:
            population_size (int): Size of the GA population.
            generations (int): Number of generations to evolve.
            cxpb (float): Crossover probability.
            mutpb (float): Mutation probability.
            seed_ratio (float): Ratio of population to seed from initial_sequence_np if available.
            verbose (bool): Whether to print evolution statistics.

        Returns:
            tuple: (best_individual_raw_format, best_fitness_mse, final_evolved_image_np)
        """
        if self.target_image_np is None:
            raise ValueError("Target image not set. Call set_target() before evolving.")

        pop = self.toolbox.population(n=population_size)

        # Seed population if initial_sequence_ga_format is available
        if self.initial_sequence_ga_format is not None and seed_ratio > 0:
            num_init_from_data = int(population_size * seed_ratio)
            for i in range(num_init_from_data):
                new_ind = creator.Individual(self.initial_sequence_ga_format)
                # Apply slight mutation to the seeded individuals to introduce diversity
                self.toolbox.mutate(new_ind, mu=0.0, sigma=0.05, indpb=0.1) 
                pop[i] = new_ind # Replace an individual in the population
        
        # Evaluate initial population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        halloffame = tools.HallOfFame(1) 

        if verbose:
            print(f"Starting SVG evolution for image size {self.image_size}...")
            print(f"Individual size: {self.INDIVIDUAL_SIZE} genes ({self.NUM_SVG_COMMANDS_IN_INDIVIDUAL} commands * {self.opts.dim_seq_short} values/command)")

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, generations, 
                                            stats=stats, halloffame=halloffame, verbose=verbose)
        if verbose:
            print("Evolution complete.")

        best_individual = halloffame[0]
        best_fitness_mse = best_individual.fitness.values[0]
        
        final_evolved_image_np = self._draw_svg_commands_on_image(best_individual, self.image_size, 
                                                                  self.util_funcs.MAX_COORD_VAL, 
                                                                  self.util_funcs.N_QUANT_BINS)
        
        return best_individual, best_fitness_mse, final_evolved_image_np

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # --- IMPORTANT: Configure your actual opts and util_funcs here if testing ---
    # For a real application, these would be imported from your main project.
    class MyActualOpts:
        def __init__(self):
            self.img_size = 64 # Match rendered_64.npy
            self.max_seq_len = 50 # Max sequence length of your SVG
            self.dim_seq_short = 9 # 1 command type + 8 args

    class MyActualUtilFuncs:
        def __init__(self):
            self.N_QUANT_BINS = 128
            self.MAX_COORD_VAL = 30.0

        def numericalize(self, coord_tensor):
            # Replace with your actual numericalize logic
            if not isinstance(coord_tensor, torch.Tensor):
                coord_tensor = torch.tensor(coord_tensor, dtype=torch.float32)
            clipped_tensor = torch.clamp(coord_tensor, 0.0, self.MAX_COORD_VAL)
            numericalized_tensor = torch.round(clipped_tensor / self.MAX_COORD_VAL * (self.N_QUANT_BINS - 1))
            return numericalized_tensor.long()

        def denumericalize(self, numerical_tensor):
            # Replace with your actual denumericalize logic
            if not isinstance(numerical_tensor, torch.Tensor):
                numerical_tensor = torch.tensor(numerical_tensor, dtype=torch.float32)
            denumericalized_tensor = numerical_tensor / (self.N_QUANT_BINS - 1) * self.MAX_COORD_VAL
            return denumericalized_tensor

    # Use your actual opts and util_funcs classes here
    my_opts = MyActualOpts()
    my_util_funcs = MyActualUtilFuncs()

    evolver = SVGEvolver(opts=my_opts, util_funcs=my_util_funcs)

    # --- Load data from .npy files for a specific glyph ---
    data_folder_path = "data/font_A" # <--- CHỈNH ĐỔI ĐƯỜNG DẪN NÀY ĐẾN THƯ MỤC CHỨA DỮ LIỆU CỦA BẠN
    
    # Example: Load 'a' character (assuming a consistent structure or selection method)
    # You might need to select a specific character's data if rendered_64.npy and sequence.npy contain multiple
    # For simplicity, assuming they are for one character if loaded directly as (H,W) or (L, D)
    
    target_img_path = os.path.join(data_folder_path, "rendered_64.npy")
    target_seq_path = os.path.join(data_folder_path, "sequence.npy") # Or sequence_relaxed.npy

    if not os.path.exists(target_img_path):
        print(f"ERROR: rendered_64.npy not found at {target_img_path}")
        print("Please check data_folder_path or create dummy data for testing.")
        # Create a dummy target image if not found for testing
        dummy_target_image = np.zeros((my_opts.img_size, my_opts.img_size), dtype=np.float32)
        ImageDraw.Draw(Image.fromarray((dummy_target_image * 255).astype(np.uint8))).ellipse((10,10,50,50), fill='white')
        dummy_target_image = np.array(Image.fromarray((dummy_target_image * 255).astype(np.uint8))) / 255.0
        evolver.set_target(dummy_target_image)
    else:
        loaded_target_image_np = np.load(target_img_path)
        loaded_target_sequence_np = None
        if os.path.exists(target_seq_path):
            loaded_target_sequence_np = np.load(target_seq_path)

        evolver.set_target(loaded_target_image_np, loaded_target_sequence_np)

    # --- Run the evolution ---
    best_individual_raw, best_mse, final_image_np = evolver.evolve_svg_for_image(
        population_size=200, 
        generations=1000, 
        cxpb=0.7, 
        mutpb=0.3,
        seed_ratio=0.1, # Use 10% of loaded sequence for seeding
        verbose=True
    )

    # --- Save results ---
    output_dir = "evolved_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save target image
    target_pil = Image.fromarray((evolver.target_image_np * 255).astype(np.uint8))
    target_pil.save(os.path.join(output_dir, "target_image.png"))

    # Save evolved image
    final_pil = Image.fromarray((final_image_np * 255).astype(np.uint8))
    final_pil.save(os.path.join(output_dir, "evolved_image.png"))

    print(f"\nResults saved to '{output_dir}' folder.")
    print(f"Best MSE: {best_mse:.6f}")

    # You can also process best_individual_raw back into numericalized format or SVG string here
    # Example: Converting best_individual_raw back to your numericalized sequence format
    # This involves reversing the scaling applied in set_target's initial_sequence_ga_format
    reconstructed_numerical_sequence = []
    for i in range(evolver.NUM_SVG_COMMANDS_IN_INDIVIDUAL):
        start_idx = i * evolver.opts.dim_seq_short
        cmd_type_raw = best_individual_raw[start_idx]
        cmd_type_numerical = int(round(cmd_type_raw * 3)) # Scale back to 0,1,2,3

        args_raw_list = best_individual_raw[start_idx + 1 : start_idx + 1 + evolver.ARGS_PER_COMMAND]
        args_numerical = [int(round(a_raw * (evolver.util_funcs.N_QUANT_BINS - 1))) for a_raw in args_raw_list]
        
        reconstructed_numerical_sequence.append([cmd_type_numerical] + args_numerical)

    reconstructed_numerical_sequence_np = np.array(reconstructed_numerical_sequence)
    print(f"\nReconstructed Numerical Sequence (first 5 commands):\n{reconstructed_numerical_sequence_np[:5]}")

    # If you have svgwrite and cairosvg installed, you can render it to a proper SVG file here
    # from your_svg_renderer_module import render_numerical_sequence_to_svg_file
    # render_numerical_sequence_to_svg_file(reconstructed_numerical_sequence_np, os.path.join(output_dir, "evolved_output.svg"))