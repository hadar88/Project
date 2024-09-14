public class FoodItem {

    private final String name;
    private final String measure; 
    private final int grams; 
    private final double calories; 
    private final double protein;
    private final double fat;
    private final double carbs;
    private final double fiber;
    //private final double sugar;
    private final String category; 

    public FoodItem(String name, String measure, int grams, double calories, double protein, double fat, double carbs, double fiber, String category) {
        this.name = name;
        this.measure = measure;
        this.grams = grams;
        this.calories = calories;
        this.protein = protein;
        this.fat = fat;
        this.carbs = carbs;
        this.fiber = fiber;
        //this.sugar = sugar;
        this.category = category;
    }

    public String getName() {
        return name;
    }

    public String getMeasure() {
        return measure;
    }

    public int getGrams() {
        return grams;
    }

    public double getCalories() {
        return calories;
    }

    public double getProtein() {
        return protein;
    }

    public double getFat() {
        return fat;
    }

    public double getCarbs() {
        return carbs;
    }

    public double getFiber() {
        return fiber;
    }

    //public double getSugar() {
    //    return sugar;
    //}

    public String getCategory() {
        return category;
    }

}
