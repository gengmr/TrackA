import sys


def process_command_args(arguments):

    # specifying default parameters

    batch_size = 8
    train_size = 512
    test_size = 16
    learning_rate = 5e-4
    num_train_iters = 300000

    w_content = 10
    w_color = 0.5
    w_texture = 1
    w_tv = 2000

    dped_dir = '/root/Desktop/SRDatasets/'
    vgg_dir = '/root/Desktop/SRDatasets/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    eval_step = 1000
    summary_step = 2

    phone = "iphone"

    PATCH_WIDTH = 512
    PATCH_HEIGHT = 512
    PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

    result_dir = '../results/'
    models_dir = '../models/'
    checkpoint_dir = models_dir

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("test_size"):
            test_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("w_content"):
            w_content = float(args.split("=")[1])

        if args.startswith("w_color"):
            w_color = float(args.split("=")[1])

        if args.startswith("w_texture"):
            w_texture = float(args.split("=")[1])

        if args.startswith("w_tv"):
            w_tv = float(args.split("=")[1])

        # -----------------------------------

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])

        if args.startswith("summary_step"):
            summary_step = int(args.split("=")[1])

        # -----------------------------------
        if args.startswith("PATCH_HEIGHT"):
            PATCH_HEIGHT = int(args.split("=")[1])

        if args.startswith("PATCH_WIDTH"):
            PATCH_WIDTH = int(args.split("=")[1])

        if args.startswith("PATCH_SIZE"):
            PATCH_SIZE = int(args.split("=")[1])

        # -----------------------------------
        if args.startswith("result_dir"):
            result_dir = args.split("=")[1]

        if args.startswith("models_dir"):
            models_dir = args.split("=")[1]

        if args.startswith("checkpoint_dir"):
            checkpoint_dir = args.split("=")[1]


    if phone == "":
        print("\nPlease specify the camera model by running the script with the following parameter:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    if phone not in ["iphone", "sony", "blackberry"]:
        print("\nPlease specify the correct camera model:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    print("\nThe following parameters will be applied for CNN training:\n")

    print("Phone model:", phone)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Training iterations:", str(num_train_iters))
    print()
    print("Content loss:", w_content)
    print("Color loss:", w_color)
    print("Texture loss:", w_texture)
    print("Total variation loss:", str(w_tv))
    print()
    print("Path to DPED dataset:", dped_dir)
    print("Path to VGG-19 network:", vgg_dir)
    print("Evaluation step:", str(eval_step))
    print()
    return phone, batch_size, train_size, test_size, learning_rate, num_train_iters, \
            w_content, w_color, w_texture, w_tv,\
            dped_dir, vgg_dir, eval_step, summary_step,\
           PATCH_WIDTH, PATCH_HEIGHT, PATCH_SIZE,\
           models_dir,result_dir,checkpoint_dir




def process_test_model_args(arguments):

    phone = ""
    dped_dir = 'dped/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the model by running the script with the following parameter:\n")
        print("python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu
