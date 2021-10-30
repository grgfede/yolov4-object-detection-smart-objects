package org.tensorflow.lite.examples.detection.utils;

public class PuntoFloating {

    private float x, y;

    public PuntoFloating(float x, float y){
        this.x = x;
        this.y = y;
    }

    public float getX(){
        return x;
    }

    public float getY() {
        return y;
    }

    public void setX(float x){
        this.x = x;
    }

    public void setY(float y){
        this.y = y;
    }
}
