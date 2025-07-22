import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import json
import threading
from pathlib import Path
import torch
import numpy as np
from standalone_operators import *

class ModelMergerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Model Merger - Advanced Operations")
        self.root.geometry("1200x800")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.loaded_models = {}
        self.operation_tree = []
        self.current_operation = None
        
        # Create main interface
        self.create_widgets()
        
    def create_widgets(self):
        # Main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Model Management and Operations
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Operation Tree and Results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
    def create_left_panel(self, parent):
        # Model Management Section
        model_frame = ttk.LabelFrame(parent, text="Model Management", padding=10)
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Load model buttons
        load_frame = ttk.Frame(model_frame)
        load_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(load_frame, text="Load PyTorch Model", 
                  command=self.load_pytorch_model).pack(side='left', padx=(0, 5))
        ttk.Button(load_frame, text="Load Safetensors", 
                  command=self.load_safetensors_model).pack(side='left', padx=(0, 5))
        ttk.Button(load_frame, text="Create Sample Model", 
                  command=self.create_sample_model).pack(side='left')
        
        # Loaded models list
        ttk.Label(model_frame, text="Loaded Models:").pack(anchor='w')
        
        models_list_frame = ttk.Frame(model_frame)
        models_list_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        self.models_listbox = tk.Listbox(models_list_frame, height=6)
        models_scrollbar = ttk.Scrollbar(models_list_frame, orient='vertical', 
                                       command=self.models_listbox.yview)
        self.models_listbox.configure(yscrollcommand=models_scrollbar.set)
        
        self.models_listbox.pack(side='left', fill='both', expand=True)
        models_scrollbar.pack(side='right', fill='y')
        
        # Operations Section
        ops_frame = ttk.LabelFrame(parent, text="Create Operations", padding=10)
        ops_frame.pack(fill='both', expand=True)
        
        # Operation type selection
        op_select_frame = ttk.Frame(ops_frame)
        op_select_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(op_select_frame, text="Operation:").pack(side='left')
        self.operation_var = tk.StringVar(value="Add")
        self.operation_combo = ttk.Combobox(op_select_frame, textvariable=self.operation_var,
                                          values=["Add", "Sub", "Multiply", "Smooth", "Extract", 
                                                "Similarities", "PowerUp", "InterpolateDifference",
                                                "TrainDiff", "WeightSumCutoff", 
                                                "ManualEnhancedInterpolateDifference",
                                                "AutoEnhancedInterpolateDifference"], 
                                          state="readonly", width=25)
        self.operation_combo.pack(side='left', padx=(5, 0))
        self.operation_combo.bind('<<ComboboxSelected>>', self.on_operation_change)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(ops_frame, text="Parameters", padding=10)
        self.params_frame.pack(fill='x', pady=(10, 0))
        
        # Create parameter widgets and update sources
        self.create_parameter_widgets()
        
        # Source selection
        sources_frame = ttk.LabelFrame(ops_frame, text="Input Sources", padding=10)
        sources_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Tensor key selection
        key_frame = ttk.Frame(sources_frame)
        key_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(key_frame, text="Tensor Key:").pack(side='left')
        self.tensor_key_var = tk.StringVar()
        self.tensor_key_combo = ttk.Combobox(key_frame, textvariable=self.tensor_key_var, 
                                           width=30)
        self.tensor_key_combo.pack(side='left', padx=(5, 0), fill='x', expand=True)
        
        # Source models selection
        self.source_vars = []
        self.source_combos = []
        self.sources_container = ttk.Frame(sources_frame)
        self.sources_container.pack(fill='both', expand=True)
        
        # Initialize with empty container
        
    def create_initial_source_selection(self):
        
        # Initialize source selection first
        self.create_initial_source_selection()
        
        # Create operation button
        create_frame = ttk.Frame(ops_frame)
        create_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(create_frame, text="Create Operation", 
                  command=self.create_operation).pack(side='left')
        ttk.Button(create_frame, text="Execute All", 
                  command=self.execute_operations).pack(side='left', padx=(10, 0))
        
    def create_right_panel(self, parent):
        # Operation Tree
        tree_frame = ttk.LabelFrame(parent, text="Operation Tree", padding=10)
        tree_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Treeview for operations
        tree_container = ttk.Frame(tree_frame)
        tree_container.pack(fill='both', expand=True)
        
        self.operation_tree_view = ttk.Treeview(tree_container, 
                                              columns=("Type", "Parameters", "Status"),
                                              show="tree headings", height=10)
        
        # Configure columns
        self.operation_tree_view.heading("#0", text="Operation")
        self.operation_tree_view.heading("Type", text="Type")
        self.operation_tree_view.heading("Parameters", text="Parameters")
        self.operation_tree_view.heading("Status", text="Status")
        
        self.operation_tree_view.column("#0", width=150)
        self.operation_tree_view.column("Type", width=100)
        self.operation_tree_view.column("Parameters", width=200)
        self.operation_tree_view.column("Status", width=80)
        
        tree_scroll = ttk.Scrollbar(tree_container, orient='vertical',
                                  command=self.operation_tree_view.yview)
        self.operation_tree_view.configure(yscrollcommand=tree_scroll.set)
        
        self.operation_tree_view.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        
        # Tree controls
        tree_controls = ttk.Frame(tree_frame)
        tree_controls.pack(fill='x', pady=(5, 0))
        
        ttk.Button(tree_controls, text="Remove Selected", 
                  command=self.remove_operation).pack(side='left')
        ttk.Button(tree_controls, text="Clear All", 
                  command=self.clear_operations).pack(side='left', padx=(5, 0))
        ttk.Button(tree_controls, text="Save Recipe", 
                  command=self.save_recipe).pack(side='left', padx=(5, 0))
        ttk.Button(tree_controls, text="Load Recipe", 
                  command=self.load_recipe).pack(side='left', padx=(5, 0))
        
        # Results/Output Section
        output_frame = ttk.LabelFrame(parent, text="Output Log", padding=10)
        output_frame.pack(fill='both', expand=True)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=12, width=50)
        self.output_text.pack(fill='both', expand=True)
        
        # Output controls
        output_controls = ttk.Frame(output_frame)
        output_controls.pack(fill='x', pady=(5, 0))
        
        ttk.Button(output_controls, text="Clear Log", 
                  command=self.clear_log).pack(side='left')
        ttk.Button(output_controls, text="Save Result Model", 
                  command=self.save_result_model).pack(side='left', padx=(10, 0))
        
    def create_parameter_widgets(self):
        # Initialize param_widgets if not exists
        if not hasattr(self, 'param_widgets'):
            self.param_widgets = {}
            
        # Clear existing widgets
        for widget in self.param_widgets.values():
            if hasattr(widget, 'destroy'):
                widget.master.destroy()  # Destroy the frame containing the widget
        self.param_widgets.clear()
        
        # Clear the params frame
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        operation = self.operation_var.get()
        
        if operation in ["Multiply"]:
            self.create_param_entry("alpha", "Alpha (multiplier):", "1.0")
        elif operation in ["Extract", "Similarities"]:
            self.create_param_entry("alpha", "Alpha:", "0.5")
            self.create_param_entry("beta", "Beta:", "0.5") 
            self.create_param_entry("gamma", "Gamma:", "1.0")
        elif operation in ["PowerUp", "InterpolateDifference", "AutoEnhancedInterpolateDifference"]:
            self.create_param_entry("alpha", "Alpha:", "0.5")
            self.create_param_entry("beta", "Beta:", "1.0")
            self.create_param_entry("gamma", "Gamma:", "0.5")
            self.create_param_entry("seed", "Seed:", "42")
        elif operation in ["ManualEnhancedInterpolateDifference"]:
            self.create_param_entry("alpha", "Alpha:", "0.5")
            self.create_param_entry("beta", "Beta (lower threshold):", "0.3")
            self.create_param_entry("gamma", "Gamma (upper threshold):", "0.7")
            self.create_param_entry("delta", "Delta (smoothness):", "0.5")
            self.create_param_entry("seed", "Seed:", "42")
        elif operation in ["WeightSumCutoff"]:
            self.create_param_entry("alpha", "Alpha:", "0.5")
            self.create_param_entry("beta", "Beta:", "0.3")
            self.create_param_entry("gamma", "Gamma:", "0.7")
    def create_initial_source_selection(self):
        """Initialize the source selection area"""
        # This will be populated when operation is selected
        pass
        
    def create_param_entry(self, param_name, label, default_value):
        frame = ttk.Frame(self.params_frame)
        frame.pack(fill='x', pady=2)
        
        ttk.Label(frame, text=label, width=20).pack(side='left')
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.pack(side='left', padx=(5, 0))
        
        self.param_widgets[param_name] = var
        
    def update_source_selection(self):
        # Clear existing source widgets
        for widget in self.sources_container.winfo_children():
            widget.destroy()
        
        self.source_vars.clear()
        self.source_combos.clear()
        
        operation = self.operation_var.get()
        
        # Determine number of sources needed
        if operation in ["Add", "Sub", "Extract", "Similarities", "PowerUp", 
                        "InterpolateDifference", "TrainDiff", "WeightSumCutoff",
                        "ManualEnhancedInterpolateDifference", "AutoEnhancedInterpolateDifference"]:
            if operation == "TrainDiff":
                num_sources = 3
                labels = ["Source A:", "Source B:", "Source C:"]
            elif operation in ["Add", "Sub", "PowerUp", "InterpolateDifference", 
                             "WeightSumCutoff", "ManualEnhancedInterpolateDifference",
                             "AutoEnhancedInterpolateDifference"]:
                num_sources = 2
                labels = ["Source A:", "Source B:"]
            elif operation in ["Extract"]:
                num_sources = 3
                labels = ["Base (optional):", "Source A:", "Source B:"]
            else:  # Similarities
                num_sources = 2
                labels = ["Source A:", "Source B:"]
        else:  # Multiply, Smooth
            num_sources = 1
            labels = ["Source:"]
        
        # Create source selection widgets
        model_names = list(self.loaded_models.keys()) + ["<None>"]
        
        for i in range(num_sources):
            frame = ttk.Frame(self.sources_container)
            frame.pack(fill='x', pady=2)
            
            ttk.Label(frame, text=labels[i], width=15).pack(side='left')
            var = tk.StringVar()
            combo = ttk.Combobox(frame, textvariable=var, values=model_names, 
                               state="readonly", width=25)
            combo.pack(side='left', padx=(5, 0))
            
            self.source_vars.append(var)
            self.source_combos.append(combo)
            
    def on_operation_change(self, event=None):
        self.create_parameter_widgets()
        self.update_tensor_keys()
        
    def update_tensor_keys(self):
        """Update tensor key combobox with available keys from loaded models"""
        all_keys = set()
        for model_tensors in self.loaded_models.values():
            all_keys.update(model_tensors.keys())
        
        self.tensor_key_combo['values'] = sorted(list(all_keys))
        
    def load_pytorch_model(self):
        file_path = filedialog.askopenfilename(
            title="Select PyTorch Model",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")]
        )
        if file_path:
            try:
                model_name = Path(file_path).stem
                checkpoint = torch.load(file_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    tensors = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    tensors = checkpoint['model']
                else:
                    tensors = checkpoint
                
                # Convert to appropriate format
                tensor_dict = {}
                for key, value in tensors.items():
                    if isinstance(value, torch.Tensor):
                        tensor_dict[key] = value
                
                load_checkpoint(model_name, tensor_dict)
                self.loaded_models[model_name] = tensor_dict
                
                self.models_listbox.insert(tk.END, model_name)
                self.log(f"Loaded PyTorch model: {model_name} ({len(tensor_dict)} tensors)")
                self.update_tensor_keys()
                self.update_source_selection()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                
    def load_safetensors_model(self):
        try:
            from safetensors import safe_open
        except ImportError:
            messagebox.showerror("Error", "safetensors library not installed. Install with: pip install safetensors")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Safetensors Model",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if file_path:
            try:
                model_name = Path(file_path).stem
                tensor_dict = {}
                
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor_dict[key] = f.get_tensor(key)
                
                load_checkpoint(model_name, tensor_dict)
                self.loaded_models[model_name] = tensor_dict
                
                self.models_listbox.insert(tk.END, model_name)
                self.log(f"Loaded Safetensors model: {model_name} ({len(tensor_dict)} tensors)")
                self.update_tensor_keys()
                self.update_source_selection()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                
    def create_sample_model(self):
        try:
            model_name = f"sample_model_{len(self.loaded_models) + 1}"
            tensor_dict = create_sample_tensors()
            
            load_checkpoint(model_name, tensor_dict)
            self.loaded_models[model_name] = tensor_dict
            
            self.models_listbox.insert(tk.END, model_name)
            self.log(f"Created sample model: {model_name} ({len(tensor_dict)} tensors)")
            self.update_tensor_keys()
            self.update_source_selection()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample model: {str(e)}")
            
    def create_operation(self):
        try:
            operation_type = self.operation_var.get()
            tensor_key = self.tensor_key_var.get()
            
            if not tensor_key:
                messagebox.showwarning("Warning", "Please select a tensor key")
                return
            
            # Get parameters
            params = {}
            for param_name, var in self.param_widgets.items():
                try:
                    if param_name == "seed":
                        params[param_name] = int(var.get())
                    else:
                        params[param_name] = float(var.get())
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {param_name}: {var.get()}")
                    return
            
            # Get sources
            sources = []
            for var in self.source_vars:
                model_name = var.get()
                if model_name and model_name != "<None>":
                    if model_name not in self.loaded_models:
                        messagebox.showerror("Error", f"Model not found: {model_name}")
                        return
                    source_op = LoadTensor(tensor_key, model_name)
                    sources.append(source_op)
                else:
                    sources.append(None)
            
            # Filter out None values for operations that don't need them
            if operation_type != "Extract":
                sources = [s for s in sources if s is not None]
            
            if not sources or (operation_type == "Extract" and len([s for s in sources if s is not None]) < 2):
                messagebox.showwarning("Warning", "Please select required source models")
                return
            
            # Create operation
            operation = self.create_operation_instance(operation_type, tensor_key, params, sources)
            
            if operation:
                self.operation_tree.append(operation)
                self.update_operation_tree()
                self.log(f"Created {operation_type} operation for {tensor_key}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create operation: {str(e)}")
            
    def create_operation_instance(self, op_type, key, params, sources):
        """Create operation instance based on type and parameters"""
        if op_type == "Add":
            return Add(key, *sources)
        elif op_type == "Sub":
            return Sub(key, *sources)
        elif op_type == "Multiply":
            return Multiply(key, params.get("alpha", 1.0), *sources)
        elif op_type == "Smooth":
            return Smooth(key, *sources)
        elif op_type == "Extract":
            return Extract(key, params.get("alpha", 0.5), params.get("beta", 0.5), 
                         params.get("gamma", 1.0), *sources)
        elif op_type == "Similarities":
            return Similarities(key, params.get("alpha", 0.5), params.get("beta", 0.5), 
                              params.get("gamma", 1.0), *sources)
        elif op_type == "PowerUp":
            return PowerUp(key, params.get("alpha", 0.5), params.get("seed", 42), *sources)
        elif op_type == "InterpolateDifference":
            return InterpolateDifference(key, params.get("alpha", 0.5), params.get("beta", 1.0),
                                       params.get("gamma", 0.5), params.get("seed", 42), *sources)
        elif op_type == "ManualEnhancedInterpolateDifference":
            return ManualEnhancedInterpolateDifference(key, params.get("alpha", 0.5), 
                                                     params.get("beta", 0.3), params.get("gamma", 0.7),
                                                     params.get("delta", 0.5), params.get("seed", 42), *sources)
        elif op_type == "AutoEnhancedInterpolateDifference":
            return AutoEnhancedInterpolateDifference(key, params.get("alpha", 0.5), 
                                                   params.get("beta", 0.3), params.get("gamma", 0.5),
                                                   params.get("seed", 42), *sources)
        elif op_type == "TrainDiff":
            return TrainDiff(key, *sources)
        elif op_type == "WeightSumCutoff":
            return WeightSumCutoff(key, params.get("alpha", 0.5), params.get("beta", 0.3),
                                 params.get("gamma", 0.7), *sources)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
            
    def update_operation_tree(self):
        """Update the operation tree view"""
        self.operation_tree_view.delete(*self.operation_tree_view.get_children())
        
        for i, operation in enumerate(self.operation_tree):
            op_name = f"op_{i+1}"
            op_type = operation.__class__.__name__
            
            # Format parameters
            params = []
            if hasattr(operation, 'alpha') and operation.alpha is not None:
                params.append(f"α={operation.alpha}")
            if hasattr(operation, 'beta') and operation.beta is not None:
                params.append(f"β={operation.beta}")
            if hasattr(operation, 'gamma') and operation.gamma is not None:
                params.append(f"γ={operation.gamma}")
            if hasattr(operation, 'delta') and operation.delta is not None:
                params.append(f"δ={operation.delta}")
            if hasattr(operation, 'seed') and operation.seed is not None:
                params.append(f"seed={operation.seed}")
            
            param_str = ", ".join(params)
            
            self.operation_tree_view.insert("", "end", iid=str(i),
                                          text=f"{op_name} ({operation.key})",
                                          values=(op_type, param_str, "Ready"))
            
    def execute_operations(self):
        """Execute all operations in the tree"""
        if not self.operation_tree:
            messagebox.showwarning("Warning", "No operations to execute")
            return
        
        def run_operations():
            self.log("Starting operation execution...")
            results = {}
            
            for i, operation in enumerate(self.operation_tree):
                try:
                    # Update status
                    self.root.after(0, lambda i=i: self.operation_tree_view.set(str(i), "Status", "Running"))
                    
                    # Execute operation
                    result = operation.merge()
                    results[f"result_{i+1}"] = result
                    
                    # Update status
                    self.root.after(0, lambda i=i: self.operation_tree_view.set(str(i), "Status", "Done"))
                    self.root.after(0, lambda op=operation, res=result: 
                                  self.log(f"Completed {op.__class__.__name__} for {op.key} - Shape: {res.shape}"))
                    
                except Exception as e:
                    self.root.after(0, lambda i=i: self.operation_tree_view.set(str(i), "Status", "Error"))
                    self.root.after(0, lambda e=e, op=operation: 
                                  self.log(f"Error in {op.__class__.__name__}: {str(e)}"))
            
            self.root.after(0, lambda: self.log(f"Execution completed. {len(results)} results generated."))
            
        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=run_operations)
        thread.daemon = True
        thread.start()
        
    def remove_operation(self):
        """Remove selected operation from tree"""
        selection = self.operation_tree_view.selection()
        if selection:
            index = int(selection[0])
            del self.operation_tree[index]
            self.update_operation_tree()
            self.log(f"Removed operation at index {index + 1}")
            
    def clear_operations(self):
        """Clear all operations"""
        self.operation_tree.clear()
        self.operation_tree_view.delete(*self.operation_tree_view.get_children())
        self.log("Cleared all operations")
        
    def save_recipe(self):
        """Save operation recipe to file"""
        if not self.operation_tree:
            messagebox.showwarning("Warning", "No operations to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Recipe",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                recipe = []
                for op in self.operation_tree:
                    op_data = {
                        "type": op.__class__.__name__,
                        "key": op.key,
                        "alpha": op.alpha,
                        "beta": op.beta,
                        "gamma": op.gamma,
                        "delta": op.delta,
                        "seed": op.seed,
                        "sources": len(op.sources)
                    }
                    recipe.append(op_data)
                
                with open(file_path, 'w') as f:
                    json.dump(recipe, f, indent=2)
                
                self.log(f"Saved recipe to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save recipe: {str(e)}")
                
    def load_recipe(self):
        """Load operation recipe from file"""
        file_path = filedialog.askopenfilename(
            title="Load Recipe",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    recipe = json.load(f)
                
                self.log(f"Loaded recipe from {file_path} ({len(recipe)} operations)")
                # Note: Full recipe loading would require reconstructing operations
                # This is a placeholder for the UI structure
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load recipe: {str(e)}")
                
    def save_result_model(self):
        """Save merged model results"""
        messagebox.showinfo("Info", "Result saving functionality would be implemented here")
        
    def clear_log(self):
        """Clear the output log"""
        self.output_text.delete(1.0, tk.END)
        
    def log(self, message):
        """Add message to output log"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelMergerGUI(root)
    root.mainloop()