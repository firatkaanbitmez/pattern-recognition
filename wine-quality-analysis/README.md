
# Wine Quality Analysis

This project is a comprehensive analysis of the Wine Quality dataset from the UCI Machine Learning Repository. The goal of this analysis is to predict wine quality based on various physicochemical properties.

## Dataset
The dataset used for this project was sourced from the UCI Machine Learning Repository, specifically the "Wine Quality Dataset." The dataset consists of two datasets:
- Red Wine
- White Wine

Both datasets include several physicochemical properties of the wines and a quality rating between 0 and 10. The quality ratings, in practice, are mostly between 3 and 9. The dataset is slightly imbalanced with more medium-quality wines and fewer high or low-quality wines.

You can access the dataset here: [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

## Project Structure
- `main.py`: The main script used for data analysis and machine learning model training.
- `Rapor.docx`: The detailed report covering the entire analysis of the dataset.
- `Sunum.pptx`: The presentation summarizing key insights and findings from the project.

## Project Analysis Steps
1. **Dataset Overview**: 
   - The wine dataset contains physicochemical tests on wine samples, including features such as acidity, chlorides, sulfur dioxide levels, alcohol content, and more.
   
2. **Data Cleaning**:
   - Checked for missing or incorrect data values.
   - No missing data was found, and all attributes were clean and ready for analysis.

3. **Statistical Analysis**:
   - Key statistics (mean, standard deviation, min, max) were calculated for each feature.
   - Box plots, histograms, and scatter plots were created to visualize the data distribution and relationships between features.

4. **Feature Engineering**:
   - Conducted correlation analysis to identify features that were most strongly correlated with wine quality.

5. **Model Training**:
   - Five classification models were tested:
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Naive Bayes
     - Random Forest
     - Decision Tree
   - The models were evaluated using metrics such as accuracy, precision, recall, and F1-score.

6. **Results**:
   - Random Forest achieved the highest accuracy for both red and white wine datasets, with an accuracy of 65.83% for red wine and 67.55% for white wine.
   - SVM and Naive Bayes performed relatively poorly due to the class imbalance in the dataset.

## Visualizations
Key visualizations included:
- Box plots and histograms for feature distributions.
- Scatter plots showing relationships between key features and wine quality.
- Violin plots highlighting the distribution of alcohol content across different quality levels.

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn

Install the required libraries with:

```bash
pip install -r requirements.txt
```

## Running the Project
To run the analysis, execute the following command:

```bash
python main.py
```

## Conclusion
This analysis shows that the physicochemical properties of wine have a significant impact on quality. The Random Forest algorithm provided the best predictive performance, especially when handling the imbalanced nature of the dataset.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Fırat Kaan Bitmez

For further information, please refer to the detailed report (`Rapor.docx`) and the presentation (`Sunum.pptx`).
