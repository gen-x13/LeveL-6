""" #genxcode - LEVEL : Handwritten number recognition """

# Import 
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from tkinter.messagebox import showinfo
from sklearn.datasets import fetch_openml
from customtkinter import filedialog as fd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------- Preparing model -------------------------#

# Load dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Preprocessing target values in standard format (int8)
mnist.keys()
mnist.target = mnist.target.astype(np.int8)

# Determining independent and dependent variable
X = np.array(mnist.data)
y = np.array(mnist.target)

# Shuffling the values of x and y on the index array
si = np.random.permutation(X.shape[0]) # Randomly permute a range (X and y)
X = X[si] # contain pictures
y = y[si] # contain labels

# Splitting data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Pipeline
model = Pipeline([
    
    ('scaler', StandardScaler()),   # Standardization of the data
    ('knn', KNeighborsClassifier()) # Neighrest neighbor detecter
    
    ])

# Parameters for KNN only
params = {
    
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance']
    
    }

# Training
grid = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=1) 
# verbose : print info of gridsearchcv
grid.fit(X_train, y_train)

# Best Parameters
params = grid.best_params_

# Performance Graph of the pipeline
results = grid.cv_results_

# Values and Mean scores from data
k_values = results['param_knn__n_neighbors'].data
mean_scores = results['mean_test_score']

# Accuracy from tested data
accuracy = grid.score(X_test, y_test)

# ------------------------------- Results & Prediction -----------------------#

# Configuration of the GUI
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib

# Results
class ShowResults(ctk.CTkScrollableFrame):
    
    def __init__(self, master, **kwargs):
        
        super().__init__(master, **kwargs)
        
        # Clean Window from widgets
        for widget in app.winfo_children():
            widget.pack_forget() # deleting widgets
        
        
        # Grid from the previous window
        self.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Return Button
        back_button = ctk.CTkButton(self, 
                                   text="← Return to menu",
                                   fg_color="grey",
                                   hover_color="black",
                                   command=lambda: self.winfo_toplevel().return_menu())
        back_button.grid(row=0, column=0, padx=10, pady=10)
            
        # Best Parameter Display
        self.show_p = ctk.CTkLabel(self, text=(f"Best parameters found : {params}"),
                                   font=("Klee One", 10), 
                                   text_color="white")
        
        self.show_p.grid(row=1, column=0, padx=50, pady=20)
        
        # Accuracy Display
        self.show_a = ctk.CTkLabel(self, text=(f"Accuracy on test set : {accuracy:.4f}"),
                                   font=("Klee One", 20), 
                                   text_color="white")
        
        self.show_a.grid(row=2, column=0, padx=50, pady=20)


        # Creation of the figure
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(k_values, mean_scores, marker='o', linestyle='-')
        ax.set_xlabel('n_neighbors')
        ax.set_ylabel('Mean Test Accuracy')
        ax.set_title('Model Performance by n_neighbors')
        ax.grid(True)

        # Display the graph
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=3, column=0, padx=1, pady=1)

# ----------------------------------The Predict Frame ------------------------#         

# Prediction Displayed after the user enter a number
class Predict(ctk.CTkScrollableFrame): 
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Clearing the window from any widgets
        for widget in app.winfo_children():
            widget.pack_forget()
        
        self.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Return Button
        back_button = ctk.CTkButton(self, 
                                   text="← Return to menu",
                                   fg_color="grey",
                                   hover_color="black",
                                   command=lambda: self.winfo_toplevel().return_menu())
        back_button.pack(pady=0)

        # Display of the prediction
        self.chat = ctk.CTkTextbox(self, height=100, state="disabled")
        self.chat.configure(fg_color="black", font=("Helvetica", 16))
        self.chat.tag_config("user", foreground="white")
        self.chat.pack(pady=(5, 5), padx=10, fill="x")
        
        # Image Prediction Frame 
        self.image_frame = ctk.CTkFrame(self, height=50)
        self.image_frame.pack(pady=3, expand=True)
        
        # Label Prediction
        self.prediction_label = ctk.CTkLabel(self, text="", font=("Helvetica", 20, "bold"))
        self.prediction_label.pack(pady=(0, 15))

        # Input Frame
        user_input_frame = ctk.CTkFrame(self)
        user_input_frame.configure(fg_color="transparent")
        user_input_frame.pack(pady=10, padx=10, fill="x")
        
        # Frame for the user input displayed
        self.input_user = ctk.CTkEntry(user_input_frame, placeholder_text="Text me...", width=550)
        self.input_user.pack(side="left", padx=5)
        
        # Sending button
        send_b = ctk.CTkButton(user_input_frame, 
                               text="Send !",
                               text_color="white",
                               fg_color="black",
                               hover_color="grey",
                               command=self.send_message)
        send_b.pack(side="left")

        self.input_user.bind("<Return>", lambda event: self.send_message())

    def send_message(self):
        user_message = self.input_user.get().strip()

        if user_message != "":
            
            self.chat.configure(state="normal")
            self.chat.insert("end", f'Number chosen: {user_message}\n', "user")
            self.chat.configure(state="disabled")
            self.chat.see("end")
            self.input_user.delete(0, "end")

            # Call prediction
            self.predict(user_message)

    def predict(self, digit):
        if digit.isdigit(): # digit is a digit
            digit = int(digit) # then we turn the digit in int
            if 0 <= digit <= 9: # if this digit is in y_test then :
                
                # Index from y_test data (examples) to be displayed for each prediction
                matches = np.where(y_test == digit)[0] 
                
                if len(matches) > 0: # If it doesn't get matches, then return an error.
                    
                    # take the first match from all examples in y_test, 
                    # we don't need all the digits existing in it.
                    # My error : [digit] but it's not what we need.
                    index = matches[0] 
                    
                    # Best estimator choosen for a better accuracy and prediction
                    best = grid.best_estimator_ 
                    
                    print(best) # Seeing the result or maybe displaying it later in the app
        
                    # Display Prediction
                    prediction = best.predict([X_test[index]])[0] # Take the match and displays it as an int not the picture type
                    self.prediction_label.configure(text=f" Model prediction: {prediction}")
                    
                    # Prediction in picture
                    image_to_display = X_test[index].reshape(28, 28)  # Original Picture to display not the numbers
                    prediction_image = image_to_display
                    
                    # Cleaning the window from previous prediction to show a new prediction.
                    for widget in self.image_frame.winfo_children():
                        widget.destroy()
                    
                    # Figure of the predicted number
                    fig, ax = plt.subplots(figsize=(2,2))
                    ax.imshow(prediction_image, cmap=matplotlib.cm.binary) # binary for the color white and black
                    ax.axis("off") # removing the axes
                    
                    canvas = FigureCanvasTkAgg(fig, master=self.image_frame)  # fig not imshow
                    canvas.draw() # drawing on the image frame
                    canvas_widget = canvas.get_tk_widget()
                    canvas_widget.pack(padx=5, pady=10)  
                    
            else:
                self.chat.configure(state="normal")
                self.chat.insert("end", "\nInvalid input! Type a number between 0 and 9.\n", "user")
                self.chat.configure(state="disabled")
                self.chat.see("end")

        else:
            self.chat.configure(state="normal")
            self.chat.insert("end", "\nInvalid input! Type a number between 0 and 9.\n", "user")
            self.chat.configure(state="disabled")
            self.chat.see("end")

# -------------------------------The Photo Recognition -----------------------#         
            
# Prediction Displayed after the user enter a number
class PhotoPredict(ctk.CTkFrame): 
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Clearing the window from any widgets
        for widget in app.winfo_children():
            widget.pack_forget()
        
        self.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Return Button
        back_button = ctk.CTkButton(self, 
                                   text="← Return to menu",
                                   fg_color="grey",
                                   hover_color="black",
                                   command=lambda: self.winfo_toplevel().return_menu())
        back_button.pack(pady=0)
        
        self.picture_setup()
        
    def picture_setup(self):
    
         self.pic_frame = ctk.CTkScrollableFrame(self)
         self.pic_frame.pack(fill="both", expand=True, padx=20, pady=10)
         
         ctk.CTkLabel(self.pic_frame, text="Select a Handwritten Number Picture", 
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
         
         def select_file():
             
             filetypes = (
                 ('png files', '*.png'),
                 ('jpg files', '*.jpg'),
                 ('jfif files', '*.jfif'),
                 ('tiff files', '*.tiff'),
                 ('bmp files', '*.bmp')
             )
         
             filename = fd.askopenfilename(
                 title='Open a file',
                 initialdir='/',
                 filetypes=filetypes)
         
             if filename:
                 self.selected_filename = filename
                 showinfo(
                     title='Selected File',
                     message=filename
                 )
                 # Traiter le fichier immédiatement
                 process_file()
             
             return filename
         
         self.p_frame = ctk.CTkScrollableFrame(self)
         self.p_frame.pack(fill="both", pady=10)
         
         self.b_frame = ctk.CTkFrame(self)
         self.b_frame.pack(pady=10)
         
         # open button
         open_button = ctk.CTkButton(
             self.b_frame,
             text='Select Picture',
             command=select_file
         )
         
         open_button.pack(side="left", padx=10)
         
         # predict button
         predict_button = ctk.CTkButton(
             self.b_frame,
             text='Predict',
             command=self.cv_image
         )
         
         predict_button.pack(side="left", padx=10)
         
         
         def process_file():
             
             if self.selected_filename is not None:
                 
                 try:
                     
                     img = Image.open(self.selected_filename)
                     self.picture = img
                     
                     self.my_image = ctk.CTkImage(light_image=img,
                                                  size=(100, 100))
                 
                     my_label = ctk.CTkLabel(self.pic_frame, text="", 
                                             image=self.my_image)
                     my_label.pack(pady=10, padx=10)
                 
                 except Exception as e:
                    
                    print(f"Error : {e}")
            
            
    def cv_image(self):
        
        self.picture = self.picture.convert("L") # convert to grayscale for morphology
        new_img = np.array(self.picture)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        target = cv.morphologyEx(new_img, cv.MORPH_GRADIENT, kernel)
        
        plt.imshow(target, cmap='gray'),plt.title('After morphology (fourth test)')

        # Hide grid lines
        plt.grid(False)
        # Hide axes ticks
        plt.axis('off')

        # Display the graph
        plt.show() 
        
        # Reshape the picture, flatten it to transform into a vector 1darray
        pic = cv.resize(target, (28, 28)).flatten().astype(np.uint8)

        
        try :
            
            best = grid.best_estimator_
            
            print(best)

            # Display Prediction
            prediction = best.predict([pic])[-1] # Predict from the picture 
            
            prediction = ctk.CTkLabel(self.p_frame, text=f"Number on the picture is : {prediction}", font=ctk.CTkFont(size=20, weight="bold")) 
            prediction.pack(pady=10, padx=10)
            
            print(f"{prediction}")
                        

        except Exception as e:
            
            nomatch = ctk.CTkLabel(self.p_frame, text=f"{e}", 
                        font=ctk.CTkFont(size=20, weight="bold"))
            nomatch.pack(pady=10, padx=10)
            print(f"No match ! Please retry! : {e}")
         
                 
        

# --------------------------------------The GUI ------------------------------#         

# Frame
class MyFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Title 
        self.label = ctk.CTkLabel(self, 
                                  text="Handwritten Recognition Menu", 
                                  font=("Klee One", 30), text_color="white")
        self.label.grid(row=1, column=0, padx=100, pady=20)
        
        # Visualizing the model's performance
        self.b_perf = ctk.CTkButton(self,
                                  text="Visualise Performance",
                                  font=("Klee One", 30), 
                                  text_color="black",
                                  fg_color="darkgrey",
                                  hover_color="white",
                                  width=400, height=32,
                                  border_width=0, 
                                  corner_radius=8,
                                  command=lambda: ShowResults(master=self.master))
        self.b_perf.grid(row=2, column=0, padx=100, pady=50)
         
        # Prediction of a handwritten number with user input
        self.b_pred = ctk.CTkButton(self,
                                  text="Choose A Number",
                                  font=("Klee One", 30), 
                                  text_color="black",
                                  fg_color="lightgrey",
                                  hover_color="darkgrey",
                                  width=400, height=32,
                                  border_width=0, 
                                  corner_radius=8,
                                  command=lambda: Predict(master=self.master))
        self.b_pred.grid(row=3, column=0, padx=100, pady=50)
        
        # Handwritten Recognition From Picture Button
        self.b_hand = ctk.CTkButton(self,
                                  text="Handwritten Picture",
                                  font=("Klee One", 30), 
                                  text_color="black",
                                  fg_color="white",
                                  hover_color="lightgrey",
                                  width=400, height=32,
                                  border_width=0, 
                                  corner_radius=8,
                                  command=lambda: PhotoPredict(master=self.master))
        self.b_hand.grid(row=4, column=0, padx=100, pady=50)
    
# ------------------------------------The App Window -------------------------#         
   
# App
class App(ctk.CTk):
    
    def __init__(self):
        
        super().__init__()
        

        self.geometry("700x600")
        self.title("Handwritten Recognition Model")
        self.grid_rowconfigure(0, weight=1)  # configure grid system
        self.grid_columnconfigure(0, weight=1)
    
        # Window leading to every other frames and functions
        self.my_frame = MyFrame(master=self, border_color="black")
        self.my_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Button destroying the principal window
        self.quit=ctk.CTkButton(self, text="Quit the app !", 
                                font=("Arial", 10), 
                                fg_color="red",
                                hover_color="black",
                                width=50, height=12,
                                border_width=0, 
                                corner_radius=8,
                                command=self.destroy)
        self.quit.grid(row=0, column=0, padx=0, pady=0, sticky="s")
    
    def return_menu(self):
        # Clear all to redraw the frame
        for widget in self.winfo_children():
            if widget != self.quit:  # Keep the quit button in the principal window
                widget.destroy()
        
        # Recreate MyFrame
        self.my_frame = MyFrame(master=self)
        self.my_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")


# Display App
app = App()
app.mainloop()
