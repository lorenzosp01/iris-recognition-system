import cv2
from src.utils.utils import drawEye
from src.data.CasiaIrisDataset import CasiaIrisDataset
import iris

if __name__=="__main__":
    # 1. Create IRISPipeline object
    iris_pipeline = iris.IRISPipeline()

    dataset = CasiaIrisDataset("F:\Dataset\Casia")
    # 2. Load IR image of an eye
    img, label = dataset.loadItem(10)

    accuracy = 0
    for i in range(dataset.__len__()):
        img, label = dataset.loadItem(i)
        output = iris_pipeline(img_data=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), eye_side="left" if "L" in label else "right")
        if output["error"] is None:
            accuracy += 1
            metadata= output["metadata"]
            img = drawEye(metadata["eye_centers"]["pupil_center"], metadata["eye_centers"]["iris_center"], img)
            #cv2.imshow("img", img)
            #print(metadata)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            #print("Eye not found")
            continue

    print(f"System accuracy: {accuracy/dataset.__len__()}")
