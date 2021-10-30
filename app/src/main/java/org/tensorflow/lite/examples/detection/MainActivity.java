package org.tensorflow.lite.examples.detection;

import androidx.annotation.NonNull;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;


public class MainActivity extends AppCompatActivity {

    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    public static Canvas canvas;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.cameraButton);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);
        txtBottom = findViewById(R.id.txtBottom);
        txtTop = findViewById(R.id.txtTop);
        txtLeft = findViewById(R.id.txtLeft);
        txtRight = findViewById(R.id.txtRight);
        txtConteggio = findViewById(R.id.txtConteggio);


        cameraButton.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivity.class)));




        detectButton.setOnClickListener(v -> {
            txtConteggio.setText("");
            txtBottom.setText("");
            txtRight.setText("");
            txtTop.setText("");
            txtLeft.setText("");
            //Paint clearPaint = new Paint();
            //clearPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
            //canvas.drawRect(0, 0, 416, 416, clearPaint);
            Handler handler = new Handler();

            new Thread(() -> {
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results);
                    }
                });
            }).start();

        });
        this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, "test1.jpg");

        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);

        this.imageView.setImageBitmap(cropBitmap);

        initBox();
    }

    private static final Logger LOGGER = new Logger();

    public static final int TF_OD_API_INPUT_SIZE = 416;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;

    private static final String TF_OD_API_MODEL_FILE = "custom_yolo4_fp16.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;

    private Classifier detector;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;

    private Button cameraButton, detectButton, changeActivity;
    private TextView txtBottom, txtTop, txtLeft, txtRight, txtConteggio;
    private ImageView imageView;

    private void initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> tracker.draw(canvas));

        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);

        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }



    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setTextSize(3 * getResources().getDisplayMetrics().density);
        paint.setStrokeWidth(1.0f);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        final ArrayList<Classifier.Recognition> attributi = new ArrayList<>();
        final ArrayList<Classifier.Recognition> smartObjects = new ArrayList<>();
        final ArrayList<Classifier.Recognition> bottiglia = new ArrayList<>();
        final ArrayList<Classifier.Recognition> pesca = new ArrayList<>();
        final ArrayList<Classifier.Recognition> chitarra = new ArrayList<>();
        final ArrayList<Classifier.Recognition> pera = new ArrayList<>();
        final ArrayList<Classifier.Recognition> postit = new ArrayList<>();


        RectF location;

        for (final Classifier.Recognition result : results) {
            location = result.getLocation();
            final int classDetected = result.getDetectedClass();

            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                Paint paint2 = new Paint();
                paint2.setColor(Color.BLUE);
                paint2.setStyle(Paint.Style.STROKE);
                paint2.setTextSize(3 * getResources().getDisplayMetrics().density);
                paint2.setStrokeWidth(1.0f);
                canvas.drawRect(location, paint);
                String title = "";
                title += result.getTitle() + " ";
                title += result.getConfidence();


                canvas.drawText(title, result.getLocation().left, result.getLocation().bottom, paint2);

                /*
                 Inserisco nelle rispettive liste gli oggetti riconosciuti
                 Oggetti attributi:
                 3 = dado
                 1 = bussola
                 4 = matita

                 Oggetti smart Objects:
                 2 = bottiglia
                 0 = pesca
                 5 = chitarra
                 6 = pera

                 POSTIT:
                 7 = postit
                */
                if (classDetected == 3){
                    attributi.add(result);
                }
                else if (classDetected==1){
                    attributi.add(result);
                }
                else if (classDetected==4){
                    attributi.add(result);
                }
                else if (classDetected == 6){
                    smartObjects.add(result);
                }
                else if (classDetected==0){
                    smartObjects.add(result);
                }
                else if (classDetected==2){
                    smartObjects.add(result);
                }
                else if (classDetected==5){
                    smartObjects.add(result);
                }
                else if (classDetected == 7){
                    postit.add(result);
                }
            }
        }
        calcolaPercorsoPostitAttributi(postit, attributi, paint);
        calcolaPercorsoAttributiSmart(attributi, smartObjects, paint);

        imageView.setImageBitmap(bitmap);
    }

    private void calcolaPercorsoAttributiSmart(ArrayList<Classifier.Recognition> attributi, ArrayList<Classifier.Recognition> smartObjects, Paint paint) {
        for (int i = 0; i < attributi.size(); i++){
            double minDist = 9999;
            float xVicino = 0, yVicino = 0;
            float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
            for (Classifier.Recognition s : smartObjects){
                x1 = attributi.get(i).getLocation().centerX();
                x2 = s.getLocation().centerX();
                y1 = attributi.get(i).getLocation().centerY();
                y2 = s.getLocation().centerY();

                double dis=Math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                if (dis < minDist){
                    minDist = dis;
                    xVicino = x2;
                    yVicino = y2;
                }
            }
            float xp = 0, yp = 0;
            xp = attributi.get(i).getLocation().centerX();
            yp = attributi.get(i).getLocation().centerY();
            canvas.drawLine(xp, yp, xVicino, yVicino, paint);
        }
    }

    private void calcolaPercorsoPostitAttributi(ArrayList<Classifier.Recognition> postit, ArrayList<Classifier.Recognition> attributi, Paint paint) {

        for (int i = 0; i < postit.size(); i++){
            double minDist = 9999;
            float xVicino = 0, yVicino = 0;
            float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
            for (Classifier.Recognition a : attributi){
                x1 = postit.get(i).getLocation().centerX();
                x2 = a.getLocation().centerX();
                y1 = postit.get(i).getLocation().centerY();
                y2 = a.getLocation().centerY();

                double dis=Math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                if (dis < minDist){
                    minDist = dis;
                    xVicino = x2;
                    yVicino = y2;
                }
            }
            float xp = 0, yp = 0;
            xp = postit.get(i).getLocation().centerX();
            yp = postit.get(i).getLocation().centerY();
            canvas.drawLine(xp, yp, xVicino, yVicino, paint);
            }
    }
}
