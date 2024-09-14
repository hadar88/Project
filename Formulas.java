public class Formulas {

    public Formulas() {
    }
    
    public int calculateBMI(double weight, int height){
        double bmi = weight / Math.pow(height/100.0, 2);
        int temp = (int)(bmi);
        if(bmi - temp >= 0.5){
            bmi = temp + 1;
        }
        else{
            bmi = temp;
        }
        return (int)bmi;
    }

    public String getBMIStatus(double bmi){
        if(bmi <= 16){
            return "Severely underweight";
        }
        else if(bmi <= 18){
            return "Underweight";
        }
        else if(bmi <= 24){
            return "Optimal weight/Healthy";
        }
        else if(bmi <= 29){
            return "Overweight";
        }
        else if(bmi <= 39){
            return "Obese";
        }
        else{
            return "Extremely obese";
        }
    }

    //public double calculateBMR(double weight, int height, int age

}
