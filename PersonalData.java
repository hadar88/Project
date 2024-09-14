public class PersonalData {
    private String name; // First name
    private int age; // In years
    private double weight; // In kilograms
    private int height; // In centimeters
    private int bmi; // Body Mass Index
    Formulas formula;

    public PersonalData(String name, int age, double weight, int height) {
        this.formula = new Formulas();
        this.name = name;
        this.age = age;
        this.weight = weight;
        this.height = height;
        this.bmi = formula.calculateBMI(weight, height);
    }

    public String getName() {
        return name;
    }

    public String setName(String name) {
        return this.name = name;
    }

    public int getAge() {
        return age;
    }

    public int setAge(int age) {
        return this.age = age;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getBmi() {
        return bmi;
    }

    public void changeBmi() {
        this.bmi = formula.calculateBMI(weight, height);
    }

    public void printData() {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
        System.out.println("Weight: " + weight);
        System.out.println("Height: " + height);
        System.out.println("BMI: " + bmi);
    }

}
