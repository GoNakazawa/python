/*
Recover limited condition in step3 from MongoDB
*/

//ボタンが押された際に、前回の情報に復元する
$(function() {
    const btn_recover = $("#btn_step3_recover");
    
    btn_recover.click(function() {

        function isEmpty(obj){
            for(let i in obj)
            {
                return false;
            }
            return true;
        }
        var temp= {"hoge": "hoge"};
        $.ajax({
            data: JSON.stringify(temp),
            type: "POST",
            url: "/recover_step3/" + exp_id,
            dataType: "json",
            contentType: "application/json"
        }).done(function(data){
            //データ受け取り
            const dataset = JSON.parse(data);
            //range
            const range_sel_vals = dataset.range.target;
            const range_lower_vals = dataset.range.range_lower;
            const range_upper_vals = dataset.range.range_upper;
            const range_clone_num = isEmpty(dataset.range.target) ? 1 : dataset.range.target.length;
            //fixed
            const fixed_sel_vals = dataset.fixed.target;
            const fixed_vals = dataset.fixed.value;
            const fixed_clone_num = isEmpty(dataset.fixed.target) ? 1 : dataset.fixed.target.length;
            //total
            const total_cb_vals = dataset.total.target;
            const total_vals = dataset.total.total;
            const total_clone_num = isEmpty(dataset.total.target) ? 1 : dataset.total.target.length;
            //combination
            const combination_cb_vals = dataset.combination.target;
            const combination_lower_vals = dataset.combination.combination_lower;
            const combination_upper_vals = dataset.combination.combination_upper;
            const combination_clone_num = isEmpty(dataset.combination.target) ? 1 : dataset.combination.target.length;
            //ratio
            const ratio1_sel_vals = dataset.ratio.target1;
            const ratio2_sel_vals = dataset.ratio.target2;
            const ratio1_vals = dataset.ratio.ratio1;
            const ratio2_vals = dataset.ratio.ratio2;
            const ratio_clone_num = isEmpty(dataset.ratio.target1) ? 1 : dataset.ratio.target1.length;
            //groupsum
            const groupsum_cb_list = new Array(5);
            const groupsum_lower_list = new Array(5);
            const groupsum_upper_list = new Array(5);
            const group_clone_list = new Array(5);
            const groupsum_total_vals = dataset.groupsum_total.total;
            const groupsum_clone_num = isEmpty(dataset.groupsum_total.total) ? 1 : dataset.groupsum_total.total.length;
            for (i=0; i<groupsum_clone_num; i++){
                groupsum_cb_list[i] = dataset.groupsum[i].target;
                groupsum_lower_list[i] = dataset.groupsum[i].lower;
                groupsum_upper_list[i] = dataset.groupsum[i].upper;
                group_clone_list[i] = dataset.groupsum[i].upper.length;
            }

            group_i = groupsum_clone_num;
            //clone groupsum
            for (i=0; i<groupsum_clone_num; i++){
                let groupsum_id = document.getElementById("id_groupsum_"+i);
                groupsum_id.style.display = "block";
                
                let div_name = "#id_groupsum_" + i + " .div_group";
                while ($(div_name).length != group_clone_list[i]){
                    if($(div_name).length < group_clone_list[i]){
                        $(div_name).first().clone(true, true).find("input:text").val("").end()
                        .find("input:checkbox").prop("checked", false).end()
                        .insertAfter($(div_name).last());
                    }
                    else if($(div_name).length > group_clone_list[i]){
                        $(div_name).last().remove();
                    }
                    else{
                        break;
                    }
                }
                for (j=0; j<group_clone_list[i]; j++){
                    $('#id_groupsum_' + i + ' .group_val_lower').eq(j).val(groupsum_lower_list[i][j]);
                    $('#id_groupsum_' + i + ' .group_val_upper').eq(j).val(groupsum_upper_list[i][j]);
                    /************************************************* */
                    if(!isEmpty(groupsum_cb_list[i]))
                    {
                        cb_len = groupsum_cb_list[i][j].length;
                        for (k=0; k<cb_len;k++)
                        {
                            if(groupsum_cb_list[i][j][k] !=""){
                                $('#id_groupsum_' + i + ' .group_cb').eq(j*cb_len + k).prop("checked", true);
                            }    
                        }
                    }
                }
                $(".group_total").eq(i).val(groupsum_total_vals[i]);

                if($(div_name).length >= 2){
                    $("#id_groupsum_" + i + " .btn_remove_group").show();
                }
                else if($(div_name).length < 2){
                    $("#id_groupsum_" + i + " .btn_remove_group").hide();
                }    
            }

            if(groupsum_clone_num >= 2){
                $("#btn_remove_groupsum").show();
            }
            else if(groupsum_clone_num < 2){
                $("#btn_remove_groupsum").hide();
            }

            //clone combination
            while ($(".div_combination").length != combination_clone_num){
                if($(".div_combination").length < combination_clone_num){
                    $(".div_combination").first().clone(true, true).find("input:text").val("").end()
                    .find("input:checkbox").prop("checked", false).end()
                    .insertAfter($(".div_combination").last());
                }
                else if($(".div_combination").length > combination_clone_num){
                    $(".div_combination").last().remove();
                }
                else{
                    break;
                }
            }
            for (i=0; i<combination_clone_num; i++){
                $(".combination_lower").eq(i).val(combination_lower_vals[i]);
                $(".combination_upper").eq(i).val(combination_upper_vals[i]);

                if(!isEmpty(combination_cb_vals))
                {
                    cb_len = combination_cb_vals[i].length;
                    for (j=0; j<cb_len;j++)
                    {
                        if(combination_cb_vals[i][j] !=""){
                            $(".combination_cb").eq(i*cb_len + j).prop("checked", true);
                        }    
                    }
                }
            }
            if($(".div_combination").length >= 2){
                $("#btn_remove_combination").show();
            }
            else if($(".div_combination").length < 2){
                $("#btn_remove_combination").hide();
            }


            //clone range
            while ($(".div_range").length != range_clone_num){
                if($(".div_range").length < range_clone_num){
                    $(".div_range").first().clone(true, true).find("input:text").val("").end()
                    .insertAfter($(".div_range").last());
                }
                else if($(".div_range").length > range_clone_num){
                    $(".div_range").last().remove();
                }
                else{
                    break;
                }
            }            
            for (i=0; i<range_clone_num; i++){
                $(".range_sel").eq(i).val(range_sel_vals[i]);
                $(".range_lower").eq(i).val(range_lower_vals[i]);
                $(".range_upper").eq(i).val(range_upper_vals[i]);
            }
            if($(".div_range").length >= 2){
                $("#btn_remove_range").show();
            }
            else if($(".div_range").length < 2){
                $("#btn_remove_range").hide();
            }

            //clone fixed
            while ($(".div_fixed").length != fixed_clone_num){
                if($(".div_fixed").length < fixed_clone_num){
                    $(".div_fixed").first().clone(true, true).find("input:text").val("").end()
                    .insertAfter($(".div_fixed").last());
                }
                else if($(".div_fixed").length > fixed_clone_num){
                    $(".div_fixed").last().remove();
                }
                else{
                    break;
                }
            }
            for (i=0; i<fixed_clone_num; i++){
                $(".fixed_sel").eq(i).val(fixed_sel_vals[i]);
                $(".fixed_val").eq(i).val(fixed_vals[i]);
            }
            if($(".div_fixed").length >= 2){
                $("#btn_remove_fixed").show();
            }
            else if($(".div_fixed").length < 2){
                $("#btn_remove_fixed").hide();
            }

            //clone total
            while ($(".div_total").length != total_clone_num){
                if($(".div_total").length < total_clone_num){
                    $(".div_total").first().clone(true, true).find("input:text").val("").end()
                    .find("input:checkbox").prop("checked", false).end()
                    .insertAfter($(".div_total").last());
                }
                else if($(".div_total").length > total_clone_num){
                    $(".div_total").last().remove();
                }
                else{
                    break;
                }
            }
            for (i=0; i<total_clone_num; i++){
                $(".total_val").eq(i).val(total_vals[i]);

                if (!isEmpty(total_cb_vals))
                {
                    cb_len = total_cb_vals[i].length;
                    for (j=0; j<cb_len;j++)
                    {
                        if(total_cb_vals[i][j] !=""){
                            $(".total_cb").eq(i*cb_len + j).prop("checked", true);
                        }
                    }
                }
            }
            if($(".div_total").length >= 2){
                $("#btn_remove_total").show();
            }
            else if($(".div_total").length < 2){
                $("#btn_remove_total").hide();
            }


            //clone ratio
            while ($(".div_ratio").length != ratio_clone_num){
                if($(".div_ratio").length < ratio_clone_num){
                    $(".div_ratio").first().clone(true, true).find("input:text").val("").end()
                    .insertAfter($(".div_ratio").last());
                }
                else if($(".div_ratio").length > ratio_clone_num){
                    $(".div_ratio").last().remove();
                }
                else{
                    break;
                }
            }
            for (i=0; i<ratio_clone_num; i++){
                $(".ratio1_sel").eq(i).val(ratio1_sel_vals[i]);
                $(".ratio2_sel").eq(i).val(ratio2_sel_vals[i]);
                $(".ratio1_val").eq(i).val(ratio1_vals[i]);
                $(".ratio2_val").eq(i).val(ratio2_vals[i]);
            }
            if($(".div_ratio").length >= 2){
                $("#btn_remove_ratio").show();
            }
            else if($(".div_ratio").length < 2){
                $("#btn_remove_ratio").hide();
            }

        });
    return false;    
    });
});