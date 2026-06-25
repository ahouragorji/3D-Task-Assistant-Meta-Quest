using UnityEngine;

public class buttonHandler : MonoBehaviour
{
    public QuestInstructionReceiver questInstructionReceiver;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
 

    // Update is called once per frame
    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.Two)){
            questInstructionReceiver.AdvanceToNextStep();
        }
        if (OVRInput.GetDown(OVRInput.Button.Four)){
            questInstructionReceiver.ReturnToPreviousStep();
        }
    }
}
