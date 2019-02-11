package com.dmp.yolov3;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.round;
import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;

public class MainActivity extends AppCompatActivity implements OnTouchListener, CvCameraViewListener2 {

    static {
        OpenCVLoader.initDebug();
    }

    private CameraBridgeViewBase mOpenCvCameraView;
    private ArrayList<String>    classes = new ArrayList<String>();
    String classesFile = "coco.names";
    String modelConfiguration = "/yolov3.cfg";
    String modelWeights = "/yolov3.weights";
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    int inpWidth = 416;
    int inpHeight = 416;
    Mat frame;
    Net net;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.yolov3cam);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
       mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setOnTouchListener(MainActivity.this);

        modelConfiguration = Environment.getExternalStorageDirectory().getPath() + modelConfiguration;
        modelWeights = Environment.getExternalStorageDirectory().getPath() + modelWeights;

        readClasses(classes, classesFile);
        net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        /*Thread thread = new Thread(){
            public void run(){
                frame = Imgcodecs.imread(Environment.getExternalStorageDirectory().getPath() + "/dog.jpg");
                Mat blob = Dnn.blobFromImage(frame, 1/255.0, new Size(inpWidth, inpHeight), new Scalar(0,0,0), true, false);
                net.setInput(blob);
                List<Mat> outs = new ArrayList<Mat>();
                net.forward(outs, getOutputsNames(net));
                postprocess(frame, outs);
                Imgcodecs.imwrite(Environment.getExternalStorageDirectory().getPath() + "/dog1.jpg", frame);
            }
        };

        thread.start();*/

    }

    private void readClasses(ArrayList<String> classes, String file){
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open(file)));

            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                classes.add(mLine);
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        return false;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }
    List<String> getOutputsNames(Net net)
    {
        ArrayList<String> names = new ArrayList<String>();
        if (names.size() == 0)
        {
            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
            //get the names of all the layers in the network
            List<String> layersNames = net.getLayerNames();

            // Get the names of the output layers in names
            for (int i = 0; i < outLayers.size(); ++i) {
                String layer = layersNames.get(outLayers.get(i).intValue()-1);
                names.add(layer);
            }
        }
        return names;
    }

    private void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat frame)
    {
        //Draw a rectangle displaying the bounding box
        rectangle(frame, new Point(left, top), new Point(right, bottom), new Scalar(255, 178, 50), 3);

        //Get the label for the class name and its confidence
        String label = String.format("%.2f", conf);
        if (classes.size() > 0)
        {
            label = classes.get(classId) + ":" + label;System.out.println(label);
        }

        //Display the label at the top of the bounding box
        int[] baseLine = new int[1];
        Size labelSize = Imgproc.getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
        top = java.lang.Math.max(top, (int)labelSize.height);
        rectangle(frame, new Point(left, top - round(1.5*labelSize.height)),
                new Point(left + round(1.5*labelSize.width), top + baseLine[0]), new Scalar(255, 255, 255), FILLED);
        putText(frame, label, new Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(0,0,0),1);
    }

    private float get_iou(Rect bb1, Rect bb2){
        int x_left = java.lang.Math.max(bb1.x, bb2.x);
        int y_top = java.lang.Math.max(bb1.y, bb2.y);
        int x_right = java.lang.Math.min(bb1.x + bb1.height, bb2.x + bb2.height);
        int y_bottom = java.lang.Math.min(bb1.y + bb1.width, bb2.y + bb2.width);
        if( x_right < x_left || y_bottom < y_top){
            return 0.0f;
        }
        int intersection_area = (x_right - x_left) * (y_bottom - y_top);
        int bb1_area = bb1.width * bb1.height;
        int bb2_area = bb2.width * bb2.height;

        float iou = intersection_area / (float)(bb1_area + bb2_area - intersection_area);

        return iou;
    }


    void postprocess(Mat frame, List<Mat> outs)
    {
        List<Integer> classIds = new ArrayList<Integer>();
        List<Float> confidences = new ArrayList<Float>();
        List<Rect> boxes = new ArrayList<Rect>();
        //List<Integer> idxs = new ArrayList<Integer>();
        List<Float> objconf = new ArrayList<Float>();

        for (int i = 0; i < outs.size(); ++i)
        {
            int cols = 0;
            for (int j = 0; j < outs.get(i).rows(); ++j)
            {
                Mat scores = outs.get(i).row(j).colRange(5, outs.get(i).row(j).cols());
                Point classIdPoint;
                double confidence;
                Core.MinMaxLocResult r = Core.minMaxLoc(scores);
                if (r.maxVal > confThreshold)
                {
                    Mat bb = outs.get(i).row(j).colRange(0, 5);
                    float[] data = new float[1];
                    bb.get(0, 0, data);

                    int centerX = (int)(data[0] * frame.cols());

                    bb.get(0, 1, data);

                    int centerY = (int)(data[0] * frame.rows());

                    bb.get(0, 2, data);

                    int width = (int)(data[0] * frame.cols());

                    bb.get(0, 3, data);

                    int height = (int)(data[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    bb.get(0, 4, data);
                    objconf.add(data[0]);

                    confidences.add((float)r.maxVal);
                    classIds.add((int)r.maxLoc.x);
                    boxes.add(new Rect(left, top, width, height));
                }
            }
        }
/*        int classesIdsSize = classIds.size();
        for (int i = 0; i < classesIdsSize; i++){
            int idx = classIds.get(0);
            if(idx != -1) {
                for (int j = i; j < classesIdsSize; j++) {
                    if (idx == classIds.get(j)) {
                        if(get_iou(boxes.get(i), boxes.get(j)) >= nmsThreshold){
                            if(objconf.get(i) > objconf.get(j)){
                                classIds.set(j, -1);
                            } else {
                                classIds.set(i, -1);
                                idx = classIds.get(j);
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < classesIdsSize; ++i)
        {
            int idx = classIds.get(i);
            if(idx != -1) {
                Rect box = boxes.get(i);
                drawPred(classIds.get(i), confidences.get(i), box.x, box.y,
                        box.x + box.width, box.y + box.height, frame);
            }
        }*/

        MatOfRect boxs =  new MatOfRect();
        boxs.fromList(boxes);
        MatOfFloat confis = new MatOfFloat();
        confis.fromList(objconf);
        MatOfInt idxs = new MatOfInt();
        Dnn.NMSBoxes(boxs, confis, confThreshold, nmsThreshold, idxs);
        if(idxs.total() > 0) {
            int[] indices = idxs.toArray();
            for (int i = 0; i < indices.length; ++i) {
                int idx = indices[i];
                Rect box = boxes.get(idx);
                drawPred(classIds.get(idx), confidences.get(idx), box.x, box.y,
                        box.x + box.width, box.y + box.height, frame);
            }
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.rgba();
        Mat dst = new Mat();
        Imgproc.cvtColor(frame, dst, Imgproc.COLOR_BGRA2BGR);
        Mat blob = Dnn.blobFromImage(dst, 1/255.0, new Size(inpWidth, inpHeight), new Scalar(0,0,0), true, false);
        net.setInput(blob);
        List<Mat> outs = new ArrayList<Mat>();
        net.forward(outs, getOutputsNames(net));
        postprocess(frame, outs);
        return frame;
    }
}
