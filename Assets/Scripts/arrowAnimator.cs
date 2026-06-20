using UnityEngine;

public class ArrowAnimator : MonoBehaviour
{
    [Header("Spin Settings")]
    public float spinSpeed = 90f; 
    public Vector3 spinAxis = new Vector3(0, 0, 1);

    [Header("Jab (Near/Far) Settings")]
    public float jabSpeed = 4f;
    public float jabDistance = 0.05f;
    public Vector3 movementAxis = new Vector3(0, 0, 1);

    // Changed to World Position
    private Vector3 _startPosition;

    void Start()
    {
        // Store the exact world position where the spawner placed us
        _startPosition = transform.position;
    }

    void Update()
    {
        // 1. Handle Spinning
        if (spinSpeed != 0)
        {
            transform.Rotate(spinAxis * (spinSpeed * Time.deltaTime), Space.Self);
        }

        if (jabSpeed != 0 && jabDistance != 0)
        {
          
            float normalizedSine = (Mathf.Sin(Time.time * jabSpeed) + 1f) / 2f;
            
            float wave = -normalizedSine * jabDistance;
            
            transform.position = _startPosition + transform.TransformDirection(movementAxis * wave);
        }
    }
}